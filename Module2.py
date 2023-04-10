import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from collections import defaultdict
from typing import Any, cast, Dict, List, Set, Tuple, Union

import numpy as np

from captum.attr import LayerActivation, LayerAttribution, LayerGradientXActivation
from captum._utils.common import _format_tensor_into_tuples, _get_module_from_name

class Concept:
    def __init__(self, id, name):
        self.id = id
        self.name = name

class TCAV:
    def __init__(self, model, layers, concepts, target):

        self.model = model
        self.layers = layers
        self.concepts = concepts
        self.target = target
        self.layer_attr_method = cast(
                LayerAttribution,
                LayerGradientXActivation(  # type: ignore
                    model, None, multiply_by_inputs=False
                ),
            )
        
    def interpret(
        self,
        inputs,
        concepts,
        target,
        additional_forward_args,
        processes,
        **kwargs,
    ) -> Dict[str, Dict[str, Dict[str, Tensor]]]:


        # Concepts are a list of all concepts
        self.compute_cavs(concepts, processes=processes) # from Ainesh


        # TO DO: Sort the concepts by length of the concept set as in the lib

        scores: Dict[str, Dict[str, Dict[str, Tensor]]] = defaultdict(
            lambda: defaultdict()
        ) # init scores dict to 0


        # Retrieves the lengths of the experimental sets so that we can sort
        # them by the length and compute TCAV scores in batches.

        # compute offsets using sorted lengths using their indices

        for layer in self.layers:
            layer_module = _get_module_from_name(self.model, layer)
            self.layer_attr_method.layer = layer_module # for sensitivity 
            attribs = self.layer_attr_method.attribute.__wrapped__(  # type: ignore
                self.layer_attr_method,  # self
                inputs,
                target=target,
                additional_forward_args=additional_forward_args,
                attribute_to_layer_input=self.attribute_to_layer_input,
                **kwargs,
            )

            attribs = _format_tensor_into_tuples(attribs)
            # n_inputs x n_features
            attribs = torch.cat(
                [torch.reshape(attrib, (attrib.shape[0], -1)) for attrib in attribs],
                dim=1,
            )

            # TO DO: handle batch computation
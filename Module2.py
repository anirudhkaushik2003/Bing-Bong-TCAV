import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import defaultdict
from typing import Any, cast, Dict, List, Set, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from captum._utils.av import AV
from captum._utils.common import _format_tensor_into_tuples, _get_module_from_name
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import LayerActivation, LayerAttribution, LayerGradientXActivation
from captum.concept._core.cav import CAV
from captum.concept._core.concept import Concept, ConceptInterpreter
from captum.concept._utils.classifier import Classifier, DefaultClassifier
from captum.concept._utils.common import concepts_to_str
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset



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
        inputs: TensorOrTupleOfTensorsGeneric,
        concept_list: List[List[Concept]],
        target: TargetType = None,
        additional_forward_args: Any = None,
        processes: int = None,
    ) -> Dict[str, Dict[str, Dict[str, Tensor]]]:
        
        # self.compute_cavs(concept_list, processes=processes)

        scores: Dict[str, Dict[str, Dict[str, Tensor]]] = defaultdict(
            lambda: defaultdict()
        )

        # Retrieves the lengths of the experimental sets so that we can sort
        # them by the length and compute TCAV scores in batches.
        len_concept_list = np.array(
            [len(i) for i in concept_list]
        )
        print(len_concept_list)
        
        concept_sorter = np.argsort(len_concept_list)

        # compute offsets using sorted lengths using their indices
        concepts_list_sort = len_concept_list[concept_sorter]
        concepts_offset_bool = [False] + list(
            concepts_list_sort[:-1] == concepts_list_sort[1:]
        )
        concepts_list_offset = []
        for i, offset in enumerate(concepts_offset_bool):
            if not offset:
                concepts_list_offset.append(i)

        concepts_list_offset.append(len(len_concept_list))

        # sort experimental sets using the length of the concepts in each set
        concept_list_sorted = np.array(concept_list, dtype=object)[
            concept_sorter
        ]

        for layer in self.layers:
            layer_module = _get_module_from_name(self.model, layer)
            self.layer_attr_method.layer = layer_module
            attribs = self.layer_attr_method.attribute.__wrapped__( 
                self.layer_attr_method,  
                inputs,
                target=target,
                additional_forward_args=additional_forward_args,
                attribute_to_layer_input=self.attribute_to_layer_input,
            )

            attribs = _format_tensor_into_tuples(attribs)
            # n_inputs x n_features
            attribs = torch.cat(
                [torch.reshape(attrib, (attrib.shape[0], -1)) for attrib in attribs],
                dim=1,
            )

            # n_experiments x n_concepts x n_features
            cavs = []
            classes = []
            for concepts in concept_list:
                concepts_key = concepts_to_str(concepts)
                cavs_stats = cast(Dict[str, Any], self.cavs[concepts_key][layer].stats)
                cavs.append(cavs_stats["weights"].float().detach().tolist())
                classes.append(cavs_stats["classes"])

            # sort cavs and classes using the length of the concepts in each set
            cavs_sorted = np.array(cavs, dtype=object)[concept_sorter]
            classes_sorted = np.array(classes, dtype=object)[concept_sorter]
            i = 0
            while i < len(concepts_list_offset) - 1:
                cav_subset = np.array(
                    cavs_sorted[concepts_list_offset[i] : concepts_list_offset[i + 1]],
                    dtype=object,
                ).tolist()
                classes_subset = classes_sorted[
                    concepts_list_offset[i] : concepts_list_offset[i + 1]
                ].tolist()

                # n_experiments x n_concepts x n_features
                cav_subset = torch.tensor(cav_subset)
                cav_subset = cav_subset.to(attribs.device)
                assert len(cav_subset.shape) == 3, (
                    "cav should have 3 dimensions: n_experiments x "
                    "n_concepts x n_features."
                )

                experimental_subset_sorted = concept_list_sorted[
                    concepts_list_offset[i] : concepts_list_offset[i + 1]
                ]
                self._tcav_sub_computation(
                    scores,
                    layer,
                    attribs,
                    cav_subset,
                    classes_subset,
                    experimental_subset_sorted,
                )
                i += 1

        return scores

    
t = TCAV(None, None, None, None)
concepts = [[Concept(1,"a",None), Concept(2,"b",None), Concept(3,"c",None)], [Concept(4,"ab",None), Concept(5,"bss",None)],[Concept(3,"c",None)]]
t.interpret(None, concepts,  None, None,None)
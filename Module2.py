import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from collections import defaultdict
from typing import Any, cast, Dict, List, Set, Tuple, Union

import numpy as np

class Concept:
    def __init__(self, id, name):
        self.id = id
        self.name = name

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

    scores: Dict[str, Dict[str, Dict[str, Tensor]]] = defaultdict(
        lambda: defaultdict()
    ) # init scores dict to 0

    # Retrieves the lengths of the experimental sets so that we can sort
    # them by the length and compute TCAV scores in batches.

    # compute offsets using sorted lengths using their indices

    for layer in self.layers:
        layer_module = _get_module_from_name(self.model, layer)
        self.layer_attr_method.layer = layer_module
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

        # n_experiments x n_concepts x n_features
        cavs = []
        classes = []
        for concepts in concepts:
            concepts_key = concepts_to_str(concepts)
            cavs_stats = cast(Dict[str, Any], self.cavs[concepts_key][layer].stats)
            cavs.append(cavs_stats["weights"].float().detach().tolist())
            classes.append(cavs_stats["classes"])

        # sort cavs and classes using the length of the concepts in each set
        cavs_sorted = np.array(cavs, dtype=object)[exp_set_lens_arg_sort]
        classes_sorted = np.array(classes, dtype=object)[exp_set_lens_arg_sort]
        i = 0
        while i < len(exp_set_offsets) - 1:
            cav_subset = np.array(
                cavs_sorted[exp_set_offsets[i] : exp_set_offsets[i + 1]],
                dtype=object,
            ).tolist()
            classes_subset = classes_sorted[
                exp_set_offsets[i] : exp_set_offsets[i + 1]
            ].tolist()

            # n_experiments x n_concepts x n_features
            cav_subset = torch.tensor(cav_subset)
            cav_subset = cav_subset.to(attribs.device)
            assert len(cav_subset.shape) == 3, (
                "cav should have 3 dimensions: n_experiments x "
                "n_concepts x n_features."
            )

            experimental_subset_sorted = concepts_sorted[
                exp_set_offsets[i] : exp_set_offsets[i + 1]
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

def _tcav_sub_computation(
    self,
    scores: Dict[str, Dict[str, Dict[str, Tensor]]],
    layer: str,
    attribs: Tensor,
    cavs: Tensor,
    classes: List[List[int]],
    concepts: List[List[Concept]],
) -> None:
    # n_inputs x n_concepts
    tcav_score = torch.matmul(attribs.float(), torch.transpose(cavs, 1, 2))
    assert len(tcav_score.shape) == 3, (
        "tcav_score should have 3 dimensions: n_experiments x "
        "n_inputs x n_concepts."
    )

    assert attribs.shape[0] == tcav_score.shape[1], (
        "attrib and tcav_score should have the same 1st and "
        "2nd dimensions respectively (n_inputs)."
    )
    # n_experiments x n_concepts
    sign_count_score = torch.mean((tcav_score > 0.0).float(), dim=1)

    magnitude_score = torch.mean(tcav_score, dim=1)

    for i, (cls_set, concepts) in enumerate(zip(classes, concepts)):
        concepts_key = concepts_to_str(concepts)

        # sort classes / concepts in the order specified in concept_keys
        concept_ord = [concept.id for concept in concepts]
        class_ord = {cls_: idx for idx, cls_ in enumerate(cls_set)}

        new_ord = torch.tensor(
            [class_ord[cncpt] for cncpt in concept_ord], device=tcav_score.device
        )

        # sort based on classes
        scores[concepts_key][layer] = {
            "sign_count": torch.index_select(
                sign_count_score[i, :], dim=0, index=new_ord
            ),
            "magnitude": torch.index_select(
                magnitude_score[i, :], dim=0, index=new_ord
            ),
        }
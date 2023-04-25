import torch
from torch.nn import Module
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os
from typing import Any, cast, Dict, List, Set, Tuple, Union
from collections import defaultdict

from captum.concept._utils.common import concepts_to_str
from captum.attr import LayerActivation, LayerAttribution, LayerGradientXActivation, LayerIntegratedGradients
from captum._utils.common import _format_tensor_into_tuples, _get_module_from_name
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.concept._utils.classifier import Classifier, DefaultClassifier
from captum._utils.av import AV

class Concept:

    def __init__(
        self, id: int, name: str, data_iter: Union[None, torch.utils.data.DataLoader]
    ) -> None:

        self.id = id
        self.name = name
        self.data_iter = data_iter

    @property
    def identifier(self) -> str:
        return "%s-%s" % (self.name, self.id)

    def __repr__(self) -> str:
        return "Concept(%r, %r)" % (self.id, self.name)
        
class CAV:
    def __init__(
        self,
        concepts: List[Concept],
        layer: str,
        stats: Dict[str, Any] = None,
        save_path: str = "./cav/",
        model_id: str = "default_model_id",
    ) -> None:

        self.concepts = concepts
        self.layer = layer
        self.stats = stats
        self.save_path = save_path
        self.model_id = model_id
        
    def assemble_save_path(
        path: str, model_id: str, concepts: List[Concept], layer: str
    ) -> str:
        file_name = concepts_to_str(concepts) + "-" + layer + ".pkl"
        return os.path.join(path, model_id, file_name)
    
    def create_cav_dir_if_missing(save_path: str, model_id: str) -> None:
        cav_model_id_path = os.path.join(save_path, model_id)
        if not os.path.exists(cav_model_id_path):
            os.makedirs(cav_model_id_path)
    
    def load(cavs_path: str, model_id: str, concepts: List[Concept], layer: str):
        cavs_path = CAV.assemble_save_path(cavs_path, model_id, concepts, layer)

        if os.path.exists(cavs_path):
            save_dict = torch.load(cavs_path)

            concept_names = save_dict["concept_names"]
            concept_ids = save_dict["concept_ids"]
            concepts = [
                Concept(concept_id, concept_name, None)
                for concept_id, concept_name in zip(concept_ids, concept_names)
            ]
            cav = CAV(concepts, save_dict["layer"], save_dict["stats"])

            return cav

        return None
    
class LabelledDataset(Dataset):
    """
    A torch Dataset whose __getitem__ returns both a batch of activation vectors,
    as well as a batch of labels associated with those activation vectors.
    It is used to train a classifier in train_tcav
    """

    def __init__(self, datasets: List[AV.AVDataset], labels: List[int]) -> None:
        """
        Creates the LabelledDataset given a list of K Datasets, and a length K
        list of integer labels representing K different concepts.
        The assumption is that the k-th Dataset of datasets is associated with
        the k-th element of labels.
        The LabelledDataset is the concatenation of the K Datasets in datasets.
        However, __get_item__ not only returns a batch of activation vectors,
        but also a batch of labels indicating which concept that batch of
        activation vectors is associated with.
        Args:
            datasets (list[Dataset]): The k-th element of datasets is a Dataset
                    representing activation vectors associated with the k-th
                    concept
            labels (list[int]): The k-th element of labels is the integer label
                    associated with the k-th concept
        """
        assert len(datasets) == len(
            labels
        ), "number of datasets does not match the number of concepts"

        from itertools import accumulate

        offsets = [0] + list(accumulate(map(len, datasets), (lambda x, y: x + y)))
        self.length = offsets[-1]
        self.datasets = datasets
        self.labels = labels
        self.lowers = offsets[:-1]
        self.uppers = offsets[1:]

    def _i_to_k(self, i):

        left, right = 0, len(self.uppers)
        while left < right:
            mid = (left + right) // 2
            if self.lowers[mid] <= i and i < self.uppers[mid]:
                return mid
            if i >= self.uppers[mid]:
                left = mid
            else:
                right = mid

    def __getitem__(self, i: int):
        """
        Returns a batch of activation vectors, as well as a batch of labels
        indicating which concept the batch of activation vectors is associated
        with.
        Args:
            i (int): which (activation vector, label) batch in the dataset to
                    return
        Returns:
            inputs (Tensor): i-th batch in Dataset (representing activation
                    vectors)
            labels (Tensor): labels of i-th batch in Dataset
        """
        assert i < self.length
        k = self._i_to_k(i)
        inputs = self.datasets[k][i - self.lowers[k]]
        assert len(inputs.shape) == 2

        labels = torch.tensor([self.labels[k]] * inputs.size(0), device=inputs.device)
        return inputs, labels

    def __len__(self) -> int:
        """
        returns the total number of batches in the labelled_dataset
        """
        return self.length

def train_cav(
    model_id,
    concepts: List[Concept],
    layers: Union[str, List[str]],
    classifier: Classifier,
    save_path: str,
    classifier_kwargs: Dict,
) -> Dict[str, Dict[str, CAV]]:

    concepts_key = concepts_to_str(concepts)
    cavs: Dict[str, Dict[str, CAV]] = defaultdict()
    cavs[concepts_key] = defaultdict()
    layers = [layers] if isinstance(layers, str) else layers
    for layer in layers:

        # Create data loader to initialize the trainer.
        datasets = [
            AV.load(save_path, model_id, concept.identifier, layer)
            for concept in concepts
        ]

        labels = [concept.id for concept in concepts]

        labelled_dataset = LabelledDataset(cast(List[AV.AVDataset], datasets), labels)

        def batch_collate(batch):
            inputs, labels = zip(*batch)
            return torch.cat(inputs), torch.cat(labels)

        dataloader = DataLoader(labelled_dataset, collate_fn=batch_collate)

        classifier_stats_dict = classifier.train_and_eval(
            dataloader, **classifier_kwargs
        )
        classifier_stats_dict = (
            {} if classifier_stats_dict is None else classifier_stats_dict
        )

        weights = classifier.weights()
        assert (
            weights is not None and len(weights) > 0
        ), "Model weights connot be None or empty"

        classes = classifier.classes()
        assert (
            classes is not None and len(classes) > 0
        ), "Classes cannot be None or empty"

        classes = (
            cast(torch.Tensor, classes).detach().numpy()
            if isinstance(classes, torch.Tensor)
            else classes
        )
        cavs[concepts_key][layer] = CAV(
            concepts,
            layer,
            {"weights": weights, "classes": classes, **classifier_stats_dict},
            save_path,
            model_id,
        )
        # Saving cavs on the disk
        cavs[concepts_key][layer].save()

    return cavs

class TCAV():

    def __init__(
        self,
        model: Module,
        layers: Union[str, List[str]],
        model_id: str = "default_model_id",
        classifier: Classifier = None,
        save_path: str = "./cav/",
        **classifier_kwargs: Any,
    ) -> None:
        self.model = model
        self.layers = [layers] if isinstance(layers, str) else layers
        self.model_id = model_id
        self.concepts: Set[Concept] = set()
        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs
        self.cavs: Dict[str, Dict[str, CAV]] = defaultdict(lambda: defaultdict())
        self.classifier = DefaultClassifier()
        self.layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False)
        

        assert model_id, (
            "`model_id` cannot be None or empty. Consider giving `model_id` "
            "a meaningful name or leave it unspecified. If model_id is unspecified we "
            "will use `default_model_id` as its default value."
        )

        self.save_path = save_path

        CAV.create_cav_dir_if_missing(self.save_path, model_id)
        
    def generate_all_activations(self) -> None:
        for concept in self.concepts:
            self.generate_activation(self.layers, concept)

    def generate_activation(self, layers: Union[str, List], concept: Concept) -> None:
        layers = [layers] if isinstance(layers, str) else layers
        layer_modules = [_get_module_from_name(self.model, layer) for layer in layers]

        layer_act = LayerActivation(self.model, layer_modules)
        assert concept.data_iter is not None, (
            "Data iterator for concept id:",
            "{} must be specified".format(concept.id),
        )
        for i, examples in enumerate(concept.data_iter):
            activations = layer_act.attribute.__wrapped__(  # type: ignore
                layer_act,
                examples,
            )
            for activation, layer_name in zip(activations, layers):
                activation = torch.reshape(activation, (activation.shape[0], -1))
                AV.save(
                    self.save_path,
                    self.model_id,
                    concept.identifier,
                    layer_name,
                    activation.detach(),
                    str(i),
                )

    def generate_activations(self, concept_layers: Dict[Concept, List[str]]) -> None:
        for concept in concept_layers:
            self.generate_activation(concept_layers[concept], concept)
        
    def load_cavs(
        self, concepts: List[Concept]
    ) -> Tuple[List[str], Dict[Concept, List[str]]]:

        concepts_key = concepts_to_str(concepts)

        layers = []
        concept_layers = defaultdict(list)

        for layer in self.layers:
            self.cavs[concepts_key][layer] = CAV.load(
                self.save_path, self.model_id, concepts, layer
            )

            # If CAV aren't loaded
            if (
                concepts_key not in self.cavs
                or layer not in self.cavs[concepts_key]
                or not self.cavs[concepts_key][layer]
            ):

                layers.append(layer)
                # For all concepts in this experimental_set
                for concept in concepts:
                    # Collect not activated layers for this concept
                    if not AV.exists(
                        self.save_path, self.model_id, layer, concept.identifier
                    ):
                        concept_layers[concept].append(layer)
        return layers, concept_layers
    
    def compute_cavs(
        self,
        experimental_sets: List[List[Concept]],
        force_train: bool = False,
        processes: int = None,
    ):

        # Update self.concepts with concepts
        for concepts in experimental_sets:
            self.concepts.update(concepts)

        concept_ids = []
        for concept in self.concepts:
            assert concept.id not in concept_ids, (
                "There is more than one instance "
                "of a concept with id {} defined in experimental sets. Please, "
                "make sure to reuse the same instance of concept".format(
                    str(concept.id)
                )
            )
            concept_ids.append(concept.id)

        if force_train:
            self.generate_all_activations()

        # List of layers per concept key (experimental_set item) to be trained
        concept_key_to_layers = defaultdict(list)

        for concepts in experimental_sets:

            concepts_key = concepts_to_str(concepts)

            # If not 'force_train', try to load a saved CAV
            if not force_train:
                layers, concept_layers = self.load_cavs(concepts)
                concept_key_to_layers[concepts_key] = layers
                # Generate activations for missing (concept, layers)
                self.generate_activations(concept_layers)
            else:
                concept_key_to_layers[concepts_key] = self.layers

        cavs_list = []
        for concepts in experimental_sets:
            cavs_list.append(
                train_cav(
                    self.model_id,
                    concepts,
                    concept_key_to_layers[concepts_to_str(concepts)],
                    cast(Classifier, self.classifier),
                    self.save_path,
                    self.classifier_kwargs,
                )
            )

        # list[Dict[concept, Dict[layer, list]]] => Dict[concept, Dict[layer, list]]
        for cavs in cavs_list:
            for c_key in cavs:
                self.cavs[c_key].update(cavs[c_key])

        return self.cavs
        

    def interpret(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        experimental_sets: List[List[Concept]],
        target: TargetType = None,
        additional_forward_args: Any = None,
        processes: int = None,
    ) -> Dict[str, Dict[str, Dict[str, Tensor]]]:
        
        self.compute_cavs(experimental_sets, processes=processes)

        scores: Dict[str, Dict[str, Dict[str, Tensor]]] = defaultdict(
            lambda: defaultdict()
        )

        # Retrieves the lengths of the experimental sets so that we can sort
        # them by the length and compute TCAV scores in batches.
        len_experimental_sets = np.array(
            [len(i) for i in experimental_sets]
        )
        print(len_experimental_sets)
        
        concept_sorter = np.argsort(len_experimental_sets)

        # compute offsets using sorted lengths using their indices
        concepts_list_sort = len_experimental_sets[concept_sorter]
        concepts_offset_bool = [False] + list(
            concepts_list_sort[:-1] == concepts_list_sort[1:]
        )
        concepts_list_offset = []
        for i, offset in enumerate(concepts_offset_bool):

            if not offset:
                concepts_list_offset.append(i)

        concepts_list_offset.append(len(len_experimental_sets))

        # sort experimental sets using the length of the concepts in each set
        experimental_sets_sorted = np.array(experimental_sets, dtype=object)[
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
                n_steps = 5
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
            for concepts in experimental_sets:
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

                experimental_subset_sorted = experimental_sets_sorted[
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

    def _tcav_sub_computation(
	self,
	scores: Dict[str, Dict[str, Dict[str, Tensor]]],
	layer: str,
	attribs: Tensor,
	cavs: Tensor,
	classes: List[List[int]],
	experimental_sets: List[List[Concept]],
) -> None:
	

        tcav_score = torch.matmul(attribs.float(), torch.transpose(cavs, 1, 2))
        
        
        assert len(tcav_score.shape) == 3
        assert attribs.shape[0] == tcav_score.shape[1]
        

        sign_count_score = torch.mean((tcav_score > 0.0).float(), dim=1)
        magnitude_score = torch.mean(tcav_score, dim=1)

        for i, (cls_set, concepts) in enumerate(zip(classes, experimental_sets)):
            concepts_key = concepts_to_str(concepts)

            concept_ord = [concept.id for concept in concepts]
            class_ord = {cls_: idx for idx, cls_ in enumerate(cls_set)}

            new_ord = torch.tensor([class_ord[cncpt] for cncpt in concept_ord], device=tcav_score.device)

            scores[concepts_key][layer] = {
                "sign_count": torch.index_select(
                    sign_count_score[i, :], dim=0, index=new_ord
                ),
                "magnitude": torch.index_select(
                    magnitude_score[i, :], dim=0, index=new_ord
                ),
            }
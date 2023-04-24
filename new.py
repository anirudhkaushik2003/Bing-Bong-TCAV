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

	assert tcav_score.ndim == 3, "Expected 3-dimensional tcav_score tensor"
	assert attribs.shape[0] == tcav_score.shape[1], "attribs and tcav_score have incompatible shapes"

	sign_count_score = (tcav_score > 0.0).float().mean(dim=1)
	magnitude_score = tcav_score.mean(dim=1)

	for i, (cls_set, concepts) in enumerate(zip(classes, experimental_sets)):
		concepts_key = concepts_to_str(concepts)

		concept_ord = torch.tensor([concept.id for concept in concepts], device=tcav_score.device)
		class_ord = {cls_: idx for idx, cls_ in enumerate(cls_set)}

		new_ord = torch.tensor([class_ord[cncpt] for cncpt in concepts], device=tcav_score.device)

		scores[concepts_key][layer] = {
			"sign_count": torch.index_select(sign_count_score[i, :], dim=0, index=new_ord),
			"magnitude": torch.index_select(magnitude_score[i, :], dim=0, index=new_ord),
		}

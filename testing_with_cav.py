import numpy as np
from cav import CAV

class Testing:
	def __init__(self,target,concepts,bottlenecks,activation_generator,alphas,random_counterpart=None,cav_dir=None,num_random_exp=5, random_concepts=None):
		self.target = target
		self.concepts = concepts
		self.bottlenecks = bottlenecks
		self.activation_generator = activation_generator
		self.alphas = alphas
		self.random_counterpart = random_counterpart
		self.cav_dir = cav_dir
		self.num_random_exp = num_random_exp
		self.random_concepts = random_concepts


	def get_direction_dir_sign(mymodel, act, cav, concept, class_id, example):
		grad = np.reshape(mymodel.get_gradient(
			act, [class_id], cav.bottleneck), -1)
		dot_prod = np.dot(grad, cav.get_direction(concept))
		return dot_prod < 0

	def compute_tcav_score(mymodel,target_class,concept,cav,class_acts,examples)
		count = 0
		class_id = mymodel.label_to_id(target_class)
		for i in range(len(class_acts)):
			act = np.expand_dims(class_acts[i], 0)
			example = examples[i]
			if Testing.get_direction_dir_sign(mymodel, act, cav, concept, class_id, example):
				count += 1
		return float(count) / float(len(class_acts))

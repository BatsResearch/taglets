import numpy as np
from .amcl_helper import compute_constraints_with_loss, Brier_loss_linear, linear_combination_labeler
from .amcl_helper import compute_constraints_with_loss2, Brier_Score_AMCL
from .amcl_helper import cross_entropy_linear, resnet_transform, logistic_regression
from .amcl_helper import projectToSimplex, projectToBall, projectCC, projectToSimplexLR
from .amcl_helper import subGradientMethod, subGradientMethod2

from weighted import UnweightedVote
from label_model import LabelModel

import pickle
import pandas as pd

class AMCLWeightedVote(UnweightedVote):
	"""
	Class representing a weighted vote of supervision sources trained through AMCL
	"""
	def __init__(self, num_classes):
		super().__init__(num_classes)
		self.theta = None
	
	def train(self, labeled_vote_matrix, labels, unlabeled_vote_matrix, cvxpy=False):
		'''
		Train a convex combination of weak supervision sources through AMCL

		Args:
		labeled_vote_matrix - outputs on labeled data (# wls, # l data, # classes)
		labels - true labels on labeled data
		unlabeled_vote_matrix - outputs on unlabeled data (# wls, # ul data, # classes)
		'''
		# pre process vote matrix
		# convert votes to one hot
		labeled_vote_matrix = np.eye(self.num_classes)[labeled_vote_matrix]
		unlabeled_vote_matrix = np.eye(self.num_classes)[unlabeled_vote_matrix]

		labeled_vote_matrix = np.transpose(labeled_vote_matrix, (1, 0, 2))
		unlabeled_vote_matrix = np.transpose(unlabeled_vote_matrix, (1, 0, 2))
		
		labels = np.eye(self.num_classes)[labels]

		# hyperparameters
		N = 4 # of wls
		eps = 0.5
		L = 2 * np.sqrt(N + 1)
		squared_diam = 2
		T = 200
		h = eps/(L*L)

		self.num_wls, num_unlab, _ = np.shape(unlabeled_vote_matrix)
		theta = np.ones(self.num_wls) * (1 / self.num_wls)

		if not cvxpy:
			constraint_matrix, constraint_vector, constraint_sign = compute_constraints_with_loss(Brier_loss_linear, 
																									unlabeled_vote_matrix, 
																									labeled_vote_matrix, 
																									labels)

			self.theta = subGradientMethod(unlabeled_vote_matrix, constraint_matrix, constraint_vector, 
											constraint_sign, Brier_loss_linear, linear_combination_labeler, 
											projectToSimplex, theta, 
											T, h, N, num_unlab, self.num_classes)

		else:
			# cvxpy implementation
			Y, constraints = compute_constraints_with_loss2(Brier_loss_linear, Brier_Score_AMCL, unlabeled_vote_matrix, 
															labeled_vote_matrix, labels)

			# print("created constraints")

			self.theta = subGradientMethod2(unlabeled_vote_matrix, Y, constraints, Brier_loss_linear, linear_combination_labeler, 
											projectToSimplex, theta, T, h, N, num_unlab, self.num_classes)

	def get_weak_labels(self, vote_matrix, *args):
		return self._get_weighted_dist(vote_matrix, self.theta)

class AMCLLogReg(LabelModel):
	"""
	Class representing a weighted vote of supervision sources trained through AMCL
	"""
	def __init__(self, num_classes):
		super().__init__(num_classes)
		self.theta = None
	
	def train(self, labeled_vote_matrix, labels, unlabeled_vote_matrix, unlabeled_data, cvxpy=False):
		'''
		Train a convex combination of weak supervision sources through AMCL on votes

		Args:
		labeled_vote_matrix - outputs on labeled data (# wls, # l data, # classes)
		labels - true labels on labeled data
		unlabeled_vote_matrix - outputs on unlabeled data (# wls, # ul data, # classes)
		'''
		# pre process vote matrix
		# convert votes to one hot
		labeled_vote_matrix = np.eye(self.num_classes)[labeled_vote_matrix]
		unlabeled_vote_matrix = np.eye(self.num_classes)[unlabeled_vote_matrix]

		labeled_vote_matrix = np.transpose(labeled_vote_matrix, (1, 0, 2))
		unlabeled_vote_matrix = np.transpose(unlabeled_vote_matrix, (1, 0, 2))
		labels = np.eye(self.num_classes)[labels]

  		# hyperparams
		N = 4 # num of wls
		eps = 0.3
		L = 2 * np.sqrt(N + 1)
		squared_diam = 2
		T = int(np.ceil(L*L*squared_diam/(eps*eps)))
		h = eps/(L*L)
		T = 200

		# assuming structure of vote matrix is (# wls, # data, # classes)
		self.num_wls, num_unlab, _ = np.shape(unlabeled_vote_matrix)
		transformed_data = resnet_transform(unlabeled_data)
		# transforming data w/ Resnet
		initial_theta = np.random.normal(0, 0.1, (len(transformed_data[0]), self.num_classes))

		if not cvxpy:
			constraint_matrix, constraint_vector, constraint_sign = compute_constraints_with_loss(cross_entropy_linear, 
																									unlabeled_vote_matrix, 
																									labeled_vote_matrix, 
																									labels)
			print("Created constraints")
			self.theta = subGradientMethod(transformed_data, constraint_matrix, constraint_vector, 
										constraint_sign, cross_entropy_linear, logistic_regression, 
										projectToBall,initial_theta, 
										T, h, N, num_unlab, self.num_classes, lr=True)
		
		else:
			Y, constraints = compute_constraints_with_loss2(Brier_loss_linear, Brier_Score_AMCL, unlabeled_vote_matrix, 
															labeled_vote_matrix, labels)
			print("Created constraints")

			self.theta = subGradientMethod2(transformed_data, Y, constraints, Brier_loss_linear, logistic_regression, 
											projectToSimplex, initial_theta, T, h, N, num_unlab, self.num_classes, lr=True)

	def get_weak_labels(self, unlabeled_data, *args):
		transformed_data = resnet_transform(unlabeled_data)
		return logistic_regression(self.theta, transformed_data)

def get_data(num):
	'''
	Function to get the data from the DARPA task
	'''

	data = pickle.load(open("./ta2.pkl", "rb"))  

	data_dict = data["Base %d" % (num)]
	df = pd.read_feather("./domain_net-clipart_labels_train.feather")	
	print("Running Base %d" % (num))
	
	l_names = data_dict["labeled_images_names"]
	l_labels = data_dict["labeled_images_labels"]
	ul_names = data_dict["unlabeled_images_names"]
	ul_votes = data_dict["unlabeled_images_votes"]
	id_class_dict = pd.Series(df["class"].values, index=df.id.values).to_dict()
	return l_names, l_labels, ul_names, ul_votes, id_class_dict

def get_test_data(num):

	data = pickle.load(open("./ta2_test_votes_full.pkl", "rb"))
	data_dict = data["Base %d" % (num)]
	df = pd.read_feather("./domain_net-clipart_labels_test.feather")

	test_names = data_dict["unlabeled_images_names"]
	test_votes = data_dict["unlabeled_images_votes"]
	id_class_dict = pd.Series(df["class"].values, index=df.id.values).to_dict()

	test_labels = [id_class_dict[x] for x in test_names]
	return test_names, test_votes, test_labels

   
def test_cc():
	'''
	Dylan's test script for evaluating AMCL (w/ convex combination of labelers + Briar score)
	Currently running on last year's DARPA eval - need to copy data to replicate
	You can change the amount of labeled data and unlabeled data by changing num_labeled_data and end_ind params.
	'''

	num_classes = 5
	labelmodel = AMCLWeightedVote(num_classes)
	
	# base = True
	# # loading last year's DARPA eval data for testing [MultiTaskModule, TransferModule, FineTuneModule, ZSLKGModule]
	# l_names, l_labels, ul_names, ul_votes, id_class_dict = get_data(1)
	# test_names, test_votes, test_labels = get_test_data(1)

	# ul_labels = [id_class_dict[x] for x in ul_names]
	# num_labeled_data = len(l_names)

	# # cutting off how much data we use
	# num_labeled_data = 400
	# end_ind = num_labeled_data + 400

	# # using the same amount of labeled data from unlabeled data since we don't have votes on original labeled data 
	# l_labels = ul_labels[:num_labeled_data]
	# l_votes = ul_votes[:num_labeled_data]
	# l_names = ul_names[:num_labeled_data]

	# ul_labels = ul_labels[num_labeled_data:end_ind]
	# ul_votes = ul_votes[num_labeled_data:end_ind]
	# ul_names = ul_names[num_labeled_data:end_ind]

	# num_unlab = len(ul_names)

	# clipart_classes = pickle.load(open("./domain_net-clipart_classes.pkl", "rb"))
	# sketch_classes = pickle.load(open("./domain_net-sketch_classes.pkl", "rb"))

	# base_class_to_ind = {x: i for i, x in enumerate(clipart_classes)}
	# adapt_class_to_ind =  {x: i for i, x in enumerate(sketch_classes)}

	# if base == 1:
	# 	l_labels = [base_class_to_ind[x] for x in l_labels]
	# 	ul_labels = [base_class_to_ind[x] for x in ul_labels]
	# else:
	# 	l_labels = [adapt_class_to_ind[x] for x in l_labels]
	# 	ul_labels = [adapt_class_to_ind[x] for x in ul_labels]

	# using the same amount of labeled data from unlabeled data since we don't have votes on original labeled data 
	l_labels = np.load("labels.npy")
	l_votes = np.load("labeled_votes.npy")
	ul_labels = np.load("unlabeled_labels.npy")
	ul_votes = np.load("unlabeled_votes.npy")
	ul_data = np.load("unlabeled_X.npy")

	l_labels = np.argmax(l_labels, axis=1)
	l_votes = np.argmax(l_votes, axis=2)
	ul_labels = np.argmax(ul_labels, axis=1)
	ul_votes = np.argmax(ul_votes, axis=2)

	l_votes = np.transpose(l_votes, (1, 0))
	ul_votes = np.transpose(ul_votes, (1, 0))

	print("Labeled Acc:")
	num_wls = len(l_votes[0])
	for i in range(num_wls):
		preds = l_votes[:,i]
		acc = np.mean(preds == l_labels)
		print("WL %d Acc: %f" % (i, acc))
	# get ind accuracies
	print("Unlabeled Acc:")
	num_wls = len(ul_votes[0])
	for i in range(num_wls):
		preds = ul_votes[:,i]
		acc = np.mean(preds == ul_labels)
		print("WL %d Acc: %f" % (i, acc))

	labelmodel.train(l_votes, l_labels, ul_votes, cvxpy=False)
	preds = labelmodel.get_weak_labels(ul_votes)
	preds = np.argmax(preds, axis=1)
	print("AMCL Acc %f" % (np.mean(preds == ul_labels)))	

	uw = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
	uw_preds = labelmodel._get_weighted_dist(ul_votes, uw)
	uw_preds = np.argmax(uw_preds, axis=1)
	print("UW Acc %f" % (np.mean(uw_preds == ul_labels)))

def test_lr():
	
	num_classes = 5
	labelmodel = AMCLLogReg(num_classes)
	base = True

	# using the same amount of labeled data from unlabeled data since we don't have votes on original labeled data 
	l_labels = np.load("labels.npy")
	l_votes = np.load("labeled_votes.npy")
	ul_labels = np.load("unlabeled_labels.npy")
	ul_votes = np.load("unlabeled_votes.npy")
	ul_data = np.load("unlabeled_X.npy")

	l_labels = np.argmax(l_labels, axis=1)
	l_votes = np.argmax(l_votes, axis=2)
	ul_labels = np.argmax(ul_labels, axis=1)
	ul_votes = np.argmax(ul_votes, axis=2)
	
	l_votes = np.transpose(l_votes, (1, 0))
	ul_votes = np.transpose(ul_votes, (1, 0))
	num_labeled_data = len(l_labels)

	labelmodel.train(l_votes, l_labels, ul_votes, ul_data, cvxpy=False)
	preds = labelmodel.get_weak_labels(ul_data)
	preds = np.argmax(preds, axis=1)

	print(preds, ul_labels)

	print("AMCL Acc %f" % (np.mean(preds == ul_labels)))
	print(np.shape(preds), np.shape(ul_labels))

	# setting to uniform weights
	uw = UnweightedVote(num_classes)
	labelmodel.theta = np.array([0.25, 0.25, 0.25, 0.25])
	uw_preds = uw.get_weak_labels(ul_votes)
	uw_preds = np.argmax(uw_preds, axis=1)
	print("UW Acc %f" % (np.mean(uw_preds == ul_labels)))

if __name__ == "__main__":
	
	test_cc()
	# test_lr()

# -----------------------------------
# Undersampling with bagging
# Author: Victoria A. Lestari
# June 12, 2017
# -----------------------------------
import numpy as np 
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.ensemble import EasyEnsemble
import collections
from sklearn.linear_model import LogisticRegression


class EnsembleClassifier:
	def __init__(self, size):
		self.models = [] # empty list for models
		self.size = size # the number of classifiers

		for i in range(size):
			model = LogisticRegression()
			self.models.append(model)

	def fit(self, X_train_agg, y_train_agg, sample_weight=None):
		print("-----------------------------------------------")
		print("X, y train agg")
		print("-----------------------------------------------")
		for i in range(self.size):
			print np.array(X_train_agg[i]).shape
			self.models[i].fit(X_train_agg[i], y_train_agg[i], sample_weight=sample_weight)

	def predict(self, X_test):
		y_preds = []

		for i in range(self.size):
			y_pred = self.models[i].predict(X_test)[0]
			y_preds.append(y_pred)

		if 0 in y_preds and 1 in y_preds:
			print("-----------------------------------------------")
			print("y preds")
			print("-----------------------------------------------")
			print y_preds

		return collections.Counter(y_preds).most_common(1)[0][0]

	def predict_proba(self, X_test):
		y_prob = [0, 0]

		for i in range(self.size):
			tmp = self.models[i].predict_proba(X_test)[0]
			y_prob[0] += tmp[0]
			y_prob[1] += tmp[1]
			
		y_prob[0] /= self.size
		y_prob[1] /= self.size

		# print "y_prob", y_prob
		return y_prob

	# ------------------------------------
	# decide the label of items given a set of labels produced by ensemble classifiers
	# ------------------------------------ 
	def aggregate_vote(self, y_aggregate):
	    ensemble_size = len(y_aggregate)
	    sample_size = len(y_aggregate[0])
	    y_decision = []

	    for i in range(sample_size):
	        tmp_y = collections.Counter([ens[i] for ens in y_aggregate]).most_common(1)[0][0]
	        y_decision.append(tmp_y)

	    return y_decision

# y_aggregate = [[1,0,0],[1,0,0],[1,1,1],[1,0,1],[0,0,0]]

# print aggregate_vote(y_aggregate)

# ------------------------------------
# X_train_agg and y_train_agg are outputs of EasyEnsemble, n sets of training data
# We build n classifiers and test them on X_test and y_test
# And then we use the function aggregate_vote to vote for the majority
# model is the model we use to train the classifier
# ------------------------------------
# def ensemble_classifier(X_train_agg, y_train_agg, X_test, y_test, model, proba=False):
# 	n = len(X_train_agg)
# 	y_test_agg = []

# 	for i in range(n):
# 		curr_X = X_train_agg[i]
# 		curr_y = y_train_agg[i]
# 		model.fit(curr_X, curr_y)

# 		y_prob = None
# 		if proba:
# 			y_prob = model.predict_proba(X_test)[0]
# 		else:
# 			y_prob = model.predict(X_test)[0]

# 		y_test_agg.append(y_prob)

# 	return aggregate_vote(y_test_agg)

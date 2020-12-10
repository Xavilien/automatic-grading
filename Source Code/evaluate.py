from initialisation import *
from models import Attention, Sum

import pickle

from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.losses import categorical_crossentropy
import pandas as pd
from tensorflow.keras.models import load_model


# Turn one hot encoding/softmax output into actual score
def score(predictions, actual):

	output_prediction = []

	# Find the highest score probability from softmax output
	for example in predictions:
		predicted_score = [0, None]
		for index, value in enumerate(example):
			if value > predicted_score[0]:
				predicted_score = [value, index]
		output_prediction.append(predicted_score[1])

	actual_score = []

	for row in actual:
		for index, value in enumerate(row):
			if value == 1:
				actual_score.append(index)

	return output_prediction, actual_score


# Get predictions based on a specific trained model and the validation set
def get_predictions(filename, i, x_valid, y_valid, att, rnn):
	filename = filename + str(i) + '.h5'

	if att == "att":
		best_model = load_model(filename, custom_objects={"Attention": Attention})
	elif rnn == "baseline":
		best_model = load_model(filename, custom_objects={"Sum": Sum})
	else:
		best_model = load_model(filename)

	predictions = best_model.predict(x_valid, verbose=0, batch_size=1, steps=None)

	output, actual = score(predictions, y_valid)

	results = {"Softmax": [predictions, y_valid], "Scores": [output, actual]}

	return results


def get_results(filename, att, rnn):
	results = []

	for i in range(KFOLDS):
		print(filename, i)

		answers, _ = load_arrays()

		# Get the training and Validation Data
		sequences = filename[7:9] + "_sequences"
		scores = filename[7:9] + "_scores"
		_, x_valid, _, y_valid = get_train_sequences(i, answers[sequences], answers[scores])

		# Get the predictions that the model gives
		results.append(get_predictions(filename, i, x_valid, y_valid, att, rnn))

	# Save the predictions so that we don't have to recalculate again
	pickle.dump(results, open(filename + "Results.pickle", "wb"))


# Generate the accuracy, loss and f1 results for a particular model
def evaluate(p):
	# Filename is where the trained models are saved
	filename = p["filename"]

	"""global count, saved, table
	if count >= saved:
		get_results(filename, p["5_att"], p["3_rnn"])"""

	results = pickle.load(open(filename + "Results.pickle", "rb"))

	accuracy = []
	loss = []
	f1 = []

	for i in range(KFOLDS):
		# Get accuracy, kappa, quadratic weighted kappa and mean error
		predictions = results[i]["Scores"][0]
		actual = results[i]["Scores"][1]

		# Calculate Accuracy, F1 and loss
		accuracy.append(accuracy_score(actual, predictions))
		f1.append(f1_score(actual, predictions, average='weighted'))
		c = np.array(categorical_crossentropy(results[i]["Softmax"][1], results[i]["Softmax"][0]))
		loss.append(sum(c) / len(c))

	# Save the results to the table that will be used to generate the CSV
	table[filename] = [accuracy, loss, f1]


if __name__ == '__main__':
	table = {}
	# count = 0
	# saved = num_models() - 200

	for params in CURR:
		evaluate(params)

	x = [["%.03f" % (sum(j)/KFOLDS) for j in table[i]] for i in table.keys()]

	r = pd.DataFrame(
		x,
		index=[" ".join(i.split('/')[1:-1]) for i in list(table.keys())],
		columns=["Accuracy", "Loss", "F1 Score"],
	)

	r.to_csv("Results (Average 2).csv")

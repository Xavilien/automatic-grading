import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model

from initialisation import GRID, KFOLDS, load_arrays, score, get_train_sequences, get_num_predictions
from models.models import Attention, Sum


FILEPATH = Path(__file__).parent.absolute()


def get_predictions(p):
	"""Generate predictions for all 5 models trained for a particular set of hyperparameters"""
	predictions = []

	for i in range(KFOLDS):
		answers, _ = load_arrays()

		# Get the training and validation data (each model has a different set due to 5-fold cross validation
		sequences = f"q{p['1_question']}_sequences"
		scores = f"q{p['1_question']}_scores"
		_, x_valid, _, y_valid = get_train_sequences(i, answers[sequences], answers[scores])

		# Load the trained model
		filename = f"{p['filename']}/{i}.h5"
		if p['5_att'] == "att":
			model = load_model(filename, custom_objects={"Attention": Attention})
		elif p['3_rnn'] == "baseline":
			model = load_model(filename, custom_objects={"Sum": Sum})
		else:
			model = load_model(filename)

		# Generate the predictions
		softmax = model.predict(x_valid, verbose=0, batch_size=1, steps=None)
		predictions.append({"Softmax": [softmax, y_valid], "Scores": [score(softmax), score(y_valid)]})

	# Save the predictions so that we don't have to recalculate again
	pickle.dump(predictions, open(p['filename'] / "predictions.pickle", "wb"))

	return predictions


def evaluate(output="mean"):
	"""
	Generate the mean/best/all accuracy, loss and f1 results for all sets of hyperparameters

	This occurs in three steps:

	1) We generate the predictions for each of the 180 models on the validation test set
	2) We evaluate the predictions of each model with the actual y labels on accuracy, loss and f1 score
	3) We compile the mean/best/all accuracy, loss and f1 results into a csv file that we save

	Similar to training, we count the number of models that we have made predictions for so that we can skip those
	whose predictions we have already made in order to save time in the event the code is interrupted.

	Parameters
		output : "mean", "best", "all"
	"""
	table = {}
	count = 1
	num_predictions = get_num_predictions()

	for p in GRID:
		filename = p["filename"]  # where the trained models have been saved

		# Step 1: Generating predictions
		# Only generate if predictions have not been generated yet
		if count > num_predictions:
			print(f"Results {count}/{len(GRID)}: {filename}")
			predictions = get_predictions(p)  # also saves results to the folder containing the 5 trained models
		else:
			predictions = pickle.load(open(filename/"predictions.pickle", "rb"))  # opens results from the folder

		# Step 2: Evaluating the predictions
		accuracy = []
		loss = []
		f1 = []

		for i in range(KFOLDS):
			scores = predictions[i]["Scores"]
			softmax = predictions[i]["Softmax"]

			# Calculate Accuracy, F1 and loss
			accuracy.append(accuracy_score(scores[1], scores[0]))
			f1.append(f1_score(scores[1], scores[0], average='weighted'))
			loss.append(np.mean(categorical_crossentropy(softmax[1], softmax[0])))

		# Save the results to the table that will be used to generate the CSV
		name = " ".join([f'q{p["1_question"]}', p["2_train"], p["3_rnn"], f'{p["4_bi"]}{p["6_emb"]}{p["5_att"]}'])
		table[name] = [accuracy, loss, f1]

		count += 1

	# Step 3: Compiling the results
	if output == "mean":
		x = [["%.03f" % np.mean(j) for j in table[i]] for i in table.keys()]
	elif output == "best":
		x = [["%.03f" % np.max(j) for j in table[i]] for i in table.keys()]
	elif output == "all":
		x = [[np.round(j, 3) for j in table[i]] for i in table.keys()]
	else:
		print("Output must either be 'mean', 'best' or 'all'")
		return

	results = pd.DataFrame(
		x,
		index=table.keys(),
		columns=["Accuracy", "Loss", "F1 Score"],
	)

	results.to_csv(FILEPATH/f"results/results_{output}.csv")


if __name__ == '__main__':
	evaluate()

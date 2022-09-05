import pickle
import numpy as np
from pathlib import Path

import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio

from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model

from initialisation import load_arrays
from models.models import Attention
from evaluate import score
from train import fit

pio.kaleido.scope.mathjax = None  # prevent an error message when saving pdf

FILEPATH = Path(__file__).parent.absolute()

# TODO: Clean up split_test.py

split = [i / 10 for i in range(1, 10)]

bigru_glove_att = {
    "1_question": ["1"],
    "2_train": ["freeze"],
    "3_rnn": ["gru"],
    "4_bi": ["bi"],
    "5_att": ["att"],
    "6_emb": ["glove"],
    "7_split": split,
    "filename": [FILEPATH / "saved_models" / "q1" / "best"],
}

bilstm_fasttext_att = {
    "1_question": ["2"],
    "2_train": ["freeze"],
    "3_rnn": ["lstm"],
    "4_bi": ["bi"],
    "5_att": ["att"],
    "6_emb": ["fasttext"],
    "7_split": split,
    "filename": [FILEPATH / "saved_models" / "q2" / "best"],
}

q1 = list(ParameterGrid(bigru_glove_att))
q2 = list(ParameterGrid(bilstm_fasttext_att))
GRID = q1 + q2


def get_train_valid(train_fraction, features, labels):
    """
    Create training and validation sequences based on the fraction of the training data we want to train on. We keep
    a constant validation set of 42 samples and use the remaining 200 samples for the training set so that it will
    divide nicely with our train_fraction.
    """

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=50)

    # Decide on number of samples for training
    train_end = int(train_fraction * 10 * 25)

    x_train = np.array([np.array(i) for i in features[:train_end]])
    x_valid = np.array([np.array(i) for i in features[-42:]])

    y_train = labels[:train_end]
    y_valid = labels[-42:]

    return x_train, x_valid, y_train, y_valid


def get_predictions(filename, qn):
    """
    Generate predictions for models trained in train().
    """
    predictions = []

    for i in split:
        print(f"{filename}/{i * 10:.0f}.h5")

        answers, embeddings = load_arrays()

        # Get the training and Validation Data
        sequences = f"q{qn}_sequences"
        scores = f"q{qn}_scores"
        _, x_valid, _, y_valid = get_train_valid(i, answers[sequences], answers[scores])

        # Load the trained model
        model = load_model(filename / f"{i * 10:.0f}.h5", custom_objects={"Attention": Attention})

        # Generate the predictions
        softmax = model.predict(x_valid, verbose=0, batch_size=1, steps=None)
        predictions.append({"Softmax": [softmax, y_valid], "Scores": [score(softmax), score(y_valid)]})

    # Save the predictions so that we don't have to recalculate again
    pickle.dump(predictions, open(filename / "predictions.pickle", "wb"))

    return predictions


def evaluate():
    """
    Generate the accuracy and f1 results for the models trained in train().
    """
    for qn, filename in enumerate(bigru_glove_att["filename"] + bilstm_fasttext_att["filename"]):
        # Skip if we have calculated the results before
        if (filename / "results.npy").exists():
            continue

        # Skip generating predictions if we have generated them before
        if (filename / "predictions.pickle").exists():
            predictions = pickle.load(open(filename / "predictions.pickle", "rb"))
        else:
            predictions = get_predictions(filename, qn + 1)

        # Calculate the accuracy and f1 score for each model
        results = []

        for i in range(9):
            scores = predictions[i]["Scores"]

            acc = accuracy_score(scores[1], scores[0])
            f1 = f1_score(scores[1], scores[0], average="weighted")

            results.append(np.array([acc, f1]))

        np.save(str(filename / "results.npy"), results)


def plot_results(mode="separate", save=False):
    """
    Plot the results generated in evaluate() into some nice graphs for visualisation.
    """
    results = [
        np.load(str(bigru_glove_att["filename"][0] / "results.npy")),
        np.load(str(bilstm_fasttext_att["filename"][0] / "results.npy"))
    ]

    x = [int(i * 10 * 25) for i in split]  # size of training set

    # Create the plots and set the layout
    acc = [go.Scatter(name=f"Accuracy (Dataset {i + 1})", x=x, y=[x[0] for x in results[i]]) for i in range(2)]
    f1 = [go.Scatter(name=f"F1 (Dataset {i + 1})", x=x, y=[x[1] for x in results[i]]) for i in range(2)]
    yaxes_range = [[0.6, 0.85], [0.0, 0.70]]
    layout = go.Layout(title=dict(xanchor="center", x=0.5),
                       xaxis=dict(title="Number of Training Responses", ticks="outside", mirror=True,
                                  linecolor="black"),
                       yaxis=dict(title="Performance", ticks="outside", mirror=True, linecolor="black"),
                       margin=dict(t=30, b=0, l=0, r=0))

    if mode == "separate":
        for i in range(2):
            fig = go.Figure(data=[acc[i], f1[i]], layout=layout)  # create the graph
            fig.update_layout(title_text=f"Dataset {i + 1} Best Model Performance against Number of Training Responses")
            fig.update_yaxes(range=yaxes_range[i])
            py.iplot(fig)
            if save:
                fig.write_image(f"Performance {i + 1}.pdf", format="pdf")

    elif mode == "together":
        fig = go.Figure(data=[acc[0], f1[0], acc[1], f1[1]], layout=layout)  # create the graph
        fig.update_layout(title_text="Performance of Best Models against Number of Training Responses")
        py.iplot(fig)
        if save:
            fig.write_image("Performance (Both).pdf", format="pdf")

    else:
        print("Invalid mode!")
        return


def train():
    """
    In order to explore how the size of the training set affects our models' performance, we vary the size of the
    training set from 25 samples to 250 samples in increments of 25 samples and train the best performing models for
    each question. We evaulate the models on the validation set which has a size of 42 samples.

    We count the number of models that have already been trained so that in the event that training is interrupted,
    the code can continue training from the last model it had been training, skipping the models that have already been
    trained.
    """
    num_models = len(list(bigru_glove_att["filename"][0].glob("*.h5"))) + \
                 len(list(bilstm_fasttext_att["filename"][0].glob("*.h5")))

    # We assume we have completed training if we have 18 trained models. If you would like to retrain the last model,
    # please delete it (note that it will also retrain the second last model as well. Alternatively, you can just
    # comment out this line
    if num_models == 18:
        return

    for i, p in enumerate(GRID):
        # Skip training if the model has already been trained
        if i < num_models - 1:
            continue

        filename = p["filename"] / f"{p['7_split'] * 10:.0f}.h5"
        print(f"Model {i + 1}/{len(GRID)}: {filename}")

        answers, embeddings = load_arrays()

        # Get the training and Validation Data
        sequences = f"q{p['1_question']}_sequences"
        scores = f"q{p['1_question']}_scores"
        x_train, x_valid, y_train, y_valid = get_train_valid(p["7_split"], answers[sequences], answers[scores])

        fit(embeddings, filename, p, x_train, x_valid, y_train, y_valid)


if __name__ == '__main__':
    train()
    evaluate()
    plot_results("separate")
    plot_results("together", save=True)

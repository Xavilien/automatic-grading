import pickle
import numpy as np
from pathlib import Path

import plotly.offline as py
import plotly.graph_objs as go

from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from initialisation import load_arrays
from models.models import get_model, Attention
from evaluate import score


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
    "filename": [FILEPATH/"saved_models"/"q1"/"best"],
}

bilstm_fasttext_att = {
    "1_question": ["2"],
    "2_train": ["freeze"],
    "3_rnn": ["lstm"],
    "4_bi": ["bi"],
    "5_att": ["att"],
    "6_emb": ["fasttext"],
    "7_split": split,
    "filename": [FILEPATH/"saved_models"/"q2"/"best"],
}

q1 = list(ParameterGrid(bigru_glove_att))
q2 = list(ParameterGrid(bilstm_fasttext_att))
GRID = q1 + q2


def create_train_valid(features, labels, train_fraction):
    """Create training and validation features and labels."""

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
    predictions = []

    for i in split:
        print(f"{filename}/{i * 10:.0f}.h5")

        answers, embeddings = load_arrays()

        # Get the training and Validation Data
        sequences = f"q{qn}_sequences"
        scores = f"q{qn}_scores"
        _, x_valid, _, y_valid = create_train_valid(answers[sequences], answers[scores], i)

        # Load the trained model
        model = load_model(filename / f"{i * 10:.0f}.h5", custom_objects={"Attention": Attention})

        # Generate the predictions
        softmax = model.predict(x_valid, verbose=0, batch_size=1, steps=None)
        predictions.append({"Softmax": [softmax, y_valid], "Scores": [score(softmax), score(y_valid)]})

    # Save the predictions so that we don't have to recalculate again
    pickle.dump(predictions, open(filename / "predictions.pickle", "wb"))

    return predictions


def evaluate():
    for qn, filename in enumerate(bigru_glove_att["filename"] + bilstm_fasttext_att["filename"]):
        if (filename / "predictions.pickle").exists():
            predictions = pickle.load(open(filename / "predictions.pickle", "rb"))
        else:
            predictions = get_predictions(filename, qn+1)

        results = []

        for i in range(9):
            scores = predictions[i]["Scores"]

            acc = accuracy_score(scores[1], scores[0])
            f1 = f1_score(scores[1], scores[0], average="weighted")

            results.append(np.array([acc, f1]))

        np.save(str(filename / "results.npy"), results)


def plot_results(save=False):
    for qn, filename in enumerate(bigru_glove_att["filename"] + bilstm_fasttext_att["filename"]):
        results = np.load(str(filename / "metrics.npy"))
        p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        x = [int(i * 10 * 25) for i in p]

        acc = go.Scatter(name="Acccuracy", x=x, y=[x[0] for x in results])
        f1 = go.Scatter(name="F1", x=x, y=[x[1] for x in results])

        title = f"Dataset {qn+1} Best Model Performance against Number of Training Responses"
        layout = go.Layout(title=dict(text=title, xanchor="center", x=0.5),
                           xaxis=dict(title="Number of Training Responses", ticks="outside", mirror=True,
                                      linecolor="black"),
                           yaxis=dict(title="Performance", ticks="outside", mirror=True, linecolor="black"),
                           margin=dict(t=30, b=0, l=0, r=0))

        fig = go.Figure(data=[acc, f1], layout=layout)

        if qn == 0:
            fig.update_yaxes(range=[0.6, 0.85])
        else:
            fig.update_yaxes(range=[0.0, 0.7])

        py.iplot(fig)

        if save:
            fig.write_image("Performance" + filename[7:9] + ".pdf", format="pdf")


def plot_results2(filename1, filename2, save=False):
    results1 = np.load(str(filename1 / "metrics.npy"))
    results2 = np.load(str(filename2 / "metrics.npy"))

    p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x = [int(i * 10 * 25) for i in p]

    acc1 = go.Scatter(name="Acccuracy (Dataset 1)", x=x, y=[x[0] for x in results1])
    f11 = go.Scatter(name="F1 (Dataset 1)", x=x, y=[x[1] for x in results1])

    acc2 = go.Scatter(name="Acccuracy (Dataset 2)", x=x, y=[x[0] for x in results2])
    f12 = go.Scatter(name="F1 (Dataset 2)", x=x, y=[x[1] for x in results2])

    title = "Performance of Best Models against Number of Training Responses"

    layout = go.Layout(title=dict(text=title, xanchor="center", x=0.5),
                       xaxis=dict(title="Number of Training Responses", ticks="outside", mirror=True,
                                  linecolor="black"),
                       yaxis=dict(title="Performance", ticks="outside", mirror=True, linecolor="black"),
                       margin=dict(t=30, b=0, l=0, r=0))

    fig = go.Figure(data=[acc1, f11, acc2, f12], layout=layout)

    # fig.update_yaxes(range=[0.6, 0.85])

    py.iplot(fig)

    if save:
        fig.write_image("Performance.pdf", format="pdf")


def train():
    num_models = len(list(bigru_glove_att["filename"][0].glob("*.h5"))) + \
                 len(list(bilstm_fasttext_att["filename"][0].glob("*.h5")))

    if num_models == 18:
        return

    for i, p in enumerate(GRID):
        if i < num_models - 1:
            continue

        filename = p["filename"] / f"{p['7_split'] * 10:.0f}.h5"
        print(f"Model {i+1}/{len(GRID)}: {filename}")

        answers, embeddings = load_arrays()

        # Get the training and Validation Data
        sequences = f"q{p['1_question']}_sequences"
        scores = f"q{p['1_question']}_scores"
        x_train, x_valid, y_train, y_valid = create_train_valid(answers[sequences], answers[scores], p["7_split"])

        model = get_model(p["2_train"] == "train",  # whether to train or freeze the embeddings
                          p["3_rnn"],  # whether to use an rnn layer or dense layer (for baseline models)
                          p["4_bi"],  # whether the rnn layer should be bidirectional
                          embeddings[f"q{p['1_question']}_{p['6_emb']}"],  # load correct embedding (glove/fasttext)
                          p["5_att"],  # whether to use an attention mechanism with rnn layer
                          p["1_question"])  # question number

        # Simple early stopping
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

        # Save the best model via checkpoints
        mc = ModelCheckpoint(filename,
                             monitor='val_accuracy',
                             mode='max',
                             verbose=1,
                             save_best_only=True)

        model.fit(x_train, y_train, epochs=30, batch_size=1,
                  validation_data=(x_valid, y_valid),
                  verbose=1,
                  callbacks=[es, mc])


if __name__ == '__main__':
    train()
    evaluate()

    filename1 = bigru_glove_att["filename"][0]
    filename2 = bilstm_fasttext_att["filename"][0]

    plot_results()
    plot_results2(filename1, filename2)

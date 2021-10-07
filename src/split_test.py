from initialisation import *
from saved_models import get_model
from evaluate import score, get_predictions

from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import pickle
from sklearn.metrics import accuracy_score, f1_score

import plotly.offline as py
import plotly.graph_objs as go

# TODO: Clean up split_test.py

split = [i / 10 for i in range(1, 10, 1)]

bigru_glove_att = {
    "question": ["1"],
    "train": ["freeze"],
    "rnn": ["gru"],
    "bi": ["bi"],
    "att": ["att"],
    "emb": ["glove"],
    "split": split,
    "filename": ["saved_models/q1/best/"],
}

bilstm_fasttext_att = {
    "question": ["2"],
    "train": ["freeze"],
    "rnn": ["lstm"],
    "bi": ["bi"],
    "att": ["att"],
    "emb": ["fasttext"],
    "split": split,
    "filename": ["saved_models/q2/best/"],
}

q1 = list(ParameterGrid(bigru_glove_att))
q2 = list(ParameterGrid(bilstm_fasttext_att))
models = q1 + q2

RANDOM_STATE = 50


def create_train_valid(features, labels, train_fraction):
    """Create training and validation features and labels."""

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * 10 * 25)

    x_train = np.array([np.array(i) for i in features[:train_end]])
    x_valid = np.array([np.array(i) for i in features[-42:]])

    y_train = labels[:train_end]
    y_valid = labels[-42:]

    return x_train, x_valid, y_train, y_valid


def run(p):
    filename = p["filename"] + str(int(p["split"] * 10))
    print(filename)

    answers, embeddings = load_arrays()

    # Get the training and Validation Data
    sequences = filename[7:9] + "_sequences"
    scores = filename[7:9] + "_scores"
    x_train, x_valid, y_train, y_valid = create_train_valid(answers[sequences], answers[scores], p["split"])

    # Load the correct embeddings
    emb = filename[7:9] + "_" + p["emb"]

    t = False  # Whether to train embeddings or not
    if p["train"] == "train":
        t = True

    model = get_model(t, p["rnn"], p["bi"], embeddings[emb], p["att"], p["question"])

    # Simple early stopping
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

    # Save the best model via checkpoints
    mc = ModelCheckpoint(filename + ".h5",
                         monitor='val_accuracy',
                         mode='max',
                         verbose=1,
                         save_best_only=True)

    model.fit(x_train, y_train, epochs=30, batch_size=1,
              validation_data=(x_valid, y_valid),
              verbose=1,
              callbacks=[es, mc])


def get_results(filename, att, rnn):
    results = []

    for i in split:
        print(filename, i)
        # notif(filename + str(i))

        answers, embeddings = load_arrays()

        # Get the training and Validation Data
        sequences = filename[7:9] + "_sequences"
        scores = filename[7:9] + "_scores"
        _, x_valid, _, y_valid = create_train_valid(answers[sequences], answers[scores], i)

        # Get the predictions that the model gives
        results.append(get_predictions(f'{filename}{int(i * 10)}.h5', x_valid, y_valid, att, rnn))

    # Save the predictions so that we don't have to recalculate again
    pickle.dump(results, open(filename + "results.pickle", "wb"))


def metrics(result):
    predictions = result[0]
    actual = result[1]

    acc = accuracy_score(actual, predictions)
    f1 = f1_score(actual, predictions, average="weighted")

    return acc, f1


def evaluate(filename):
    results = pickle.load(open(filename + "results.pickle", "rb"))

    for i in range(9):
        pass

    evaluation = []

    for i in range(9):
        result = results[i]["Scores"]

        acc, f1 = metrics(result)

        evaluation.append(np.array([acc, f1]))

    evaluation = np.array(evaluation)
    np.save(filename + "metrics.npy", evaluation)
    plot_results(filename)


def evaluate2(filename):
    results = pickle.load(open(filename + "results.pickle", "rb"))
    results1 = pickle.load(open(filename + "Results1.pickle", "rb"))

    for i in range(9):
        pass

    evaluation = []

    for i in range(9):
        result = results[i]["Scores"]
        result1 = results1[i]["Scores"]

        acc, f1 = metrics(result)
        acc1, f11 = metrics(result1)

        if acc > acc1:
            evaluation.append(np.array([acc, f1]))
        else:
            evaluation.append(np.array([acc1, f11]))

    plot_results(filename, evaluation)


def plot_results(filename):
    results = np.load(filename + "metrics.npy")
    p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x = [int(i * 10 * 25) for i in p]

    acc = go.Scatter(name="Acccuracy", x=x, y=[x[0] for x in results])
    f1 = go.Scatter(name="F1", x=x, y=[x[1] for x in results])

    title = "Dataset %s Best Model Performance against Number of Training Responses" % filename[8]
    layout = go.Layout(title=dict(text=title, xanchor="center", x=0.5),
                       xaxis=dict(title="Number of Training Responses", ticks="outside", mirror=True,
                                  linecolor="black"),
                       yaxis=dict(title="Performance", ticks="outside", mirror=True, linecolor="black"),
                       margin=dict(t=30, b=0, l=0, r=0))

    fig = go.Figure(data=[acc, f1], layout=layout)

    if filename[7:9] == "q1":
        fig.update_yaxes(range=[0.6, 0.85])
    else:
        fig.update_yaxes(range=[0.0, 0.7])

    fig.write_image("Performance" + filename[7:9] + ".pdf", format="pdf")
    py.iplot(fig)


def plot_results2(filename1, filename2):
    results1 = np.load(filename1 + "metrics.npy")
    results2 = np.load(filename2 + "metrics.npy")

    p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x = [int(i * 10 * 25) for i in p]

    acc1 = go.Scatter(name="Acccuracy (Dataset 1)", x=x, y=[x[0] for x in results1])
    f11 = go.Scatter(name="F1 (Dataset 1)", x=x, y=[x[1] for x in results1])

    acc2 = go.Scatter(name="Acccuracy (Dataset 2)", x=x, y=[x[0] for x in results2])
    f12 = go.Scatter(name="F1 (Dataset 2)", x=x, y=[x[1] for x in results2])

    title = "Performance of Best saved_models against Number of Training Responses"

    layout = go.Layout(title=dict(text=title, xanchor="center", x=0.5),
                       xaxis=dict(title="Number of Training Responses", ticks="outside", mirror=True,
                                  linecolor="black"),
                       yaxis=dict(title="Performance", ticks="outside", mirror=True, linecolor="black"),
                       margin=dict(t=30, b=0, l=0, r=0))

    fig = go.Figure(data=[acc1, f11, acc2, f12], layout=layout)

    # fig.update_yaxes(range=[0.6, 0.85])

    fig.write_image("Performance.pdf", format="pdf")
    # py.iplot(fig)


def check_scores():
    answers, _ = load_arrays()

    x_train, x_valid, y_train, y_valid = create_train_valid(answers["q1_sequences"], answers["q1_scores"], 0.1)
    x = score(y_valid)
    for i in range(3):
        print(x.count(i))


if __name__ == '__main__':
    filename1 = "saved_models/q1/best/"
    filename2 = "saved_models/q2/best/"

    saved = num_models([{"filename": filename1}]) + num_models([{"filename": filename2}]) - 1

    """for i, m in enumerate(saved_models):
        if i < saved:
            continue

        run(m)
        notif(i + 1)"""

    # get_results(filename1, "att", "gru")
    # evaluate(filename1)
    # get_results(filename2, "att", "lstm")
    # evaluate(filename2)

    # plot_results(filename1)
    # plot_results(filename2)
    plot_results2(filename1, filename2)

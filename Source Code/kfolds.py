from initialisation import *
from models import get_model
from evaluate import score, get_predictions

import pickle
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.losses import categorical_crossentropy

bigru_glove_att = {
    "question": "1",
    "train": "freeze",
    "rnn": "gru",
    "bi": "bi",
    "att": "att",
    "emb": "glove",
    "filename": "Models/q1/kfolds/",
}

bilstm_fasttext_att = {
    "question": "2",
    "train": "freeze",
    "rnn": "lstm",
    "bi": "bi",
    "att": "att",
    "emb": "fasttext",
    "filename": "Models/q2/kfolds/",
}

models = [bigru_glove_att, bilstm_fasttext_att]

RANDOM_STATE = 50


def run(p):
    if num_models([p]) >= 5:
        return

    for i in range(KFOLDS):
        if i < num_models([p]) - 1:
            continue

        filename = p["filename"] + str(i)
        print(filename)

        answers, embeddings = load_arrays()

        # Get the training and Validation Data
        sequences = filename[7:9] + "_sequences"
        Models / q1 / freeze / baseline / Results.pickle   scores = filename[7:9] + "_scores"
        x_train, x_valid, y_train, y_valid = get_train_sequences(i, answers[sequences], answers[scores])

        # Load the correct embeddings
        emb = filename[7:9] + "_" + p["emb"]

        t = False  # Whether to train embeddings or not
        if p["train"] == "train":
            t = True

        m = get_model(t, p["rnn"], p["bi"], embeddings[emb], p["att"], p["question"])

        m.fit(x_train, y_train, epochs=10, batch_size=1,
              validation_data=(x_valid, y_valid),
              verbose=1, )

        m.save(filename + '.h5')


def get_results(p):
    filename = p["filename"]
    att = p["att"]
    rnn = p["rnn"]

    results = []

    for i in range(KFOLDS):
        print(filename, i)

        answers, _ = load_arrays()

        # Get the training and Validation Data
        sequences = filename[7:9] + "_sequences"
        scores = filename[7:9] + "_scores"
        _, x_valid, _, y_valid = get_train_sequences(i, answers[sequences], answers[scores])

        # Get the predictions that the model gives
        results.append(get_predictions(filename, str(i), x_valid, y_valid, att, rnn))

    # Save the predictions so that we don't have to recalculate again
    pickle.dump(results, open(filename + "Results.pickle", "wb"))


def metrics(results):
    predictions = results["Scores"][0]
    actual = results["Scores"][1]

    acc = accuracy_score(actual, predictions)
    f1 = f1_score(actual, predictions, average="weighted")
    c = np.array(categorical_crossentropy(results["Softmax"][1], results["Softmax"][0]))
    loss = sum(c) / len(c)

    return acc, f1, loss


def evaluate(p):
    filename = p["filename"]

    if num_models([p]) == KFOLDS + 2:
        return
    elif num_models([p]) < KFOLDS + 1:
        get_results(model)

    results = pickle.load(open(filename + "Results.pickle", "rb"))

    evaluation = dict(
        acc=[],
        f1=[],
        loss=[]
    )

    for i in range(KFOLDS):
        acc, f1, loss = metrics(results[i])

        evaluation["acc"].append(acc)
        evaluation["f1"].append(f1)
        evaluation["loss"].append(loss)

    pickle.dump(evaluation, open(filename + "metrics.pickle", "wb"))


def show_results(p):
    filename = p["filename"]
    results = pickle.load(open(filename + "metrics.pickle", "rb"))
    print(filename)
    print("Accuracy: %.03f" % (np.mean(results["acc"])))
    print("F1: %.03f" % (np.mean(results["f1"])))
    print("Loss: %.03f" % (np.mean(results["loss"])))


def check_scores():
    x_train, x_valid, y_train, y_valid = get_train_sequences(answers["q1_sequences"], answers["q1_scores"], 0.1)
    x, _ = score(y_valid, y_valid)
    for i in range(3):
        print(x.count(i))


if __name__ == '__main__':
    filename1 = "Models/q1/kfolds/"
    filename2 = "Models/q2/kfolds/"

    saved = num_models([{"filename": filename1}]) + num_models([{"filename": filename2}]) - 1

    for model in models:
        run(model)
        evaluate(model)
        show_results(model)

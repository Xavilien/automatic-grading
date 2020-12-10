import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from evaluate import score
import os


def load_arrays():
    """Return the pre-saved arrays"""
    # Student answers with the corresponding scores
    answers = dict(
        q1_answers=np.load("Arrays/answers.npy", allow_pickle=True),
        q1_scores=np.load("Arrays/scores.npy"),
        q1_sequences=np.load("Arrays/sequences.npy"),

        q2_answers=np.load("Arrays2/answers.npy", allow_pickle=True),
        q2_scores=np.load("Arrays2/scores.npy"),
        q2_sequences=np.load("Arrays2/sequences.npy"),
    )

    # GloVe, fastText and LDA embeddings
    embeddings = dict(
        q1_glove=np.load("Arrays/embedding_matrix.npy"),
        q1_fasttext=np.load("Arrays/embedding_matrix_fasttext.npy"),
        q1_lda=np.load("Arrays/embedding_matrix_lda.npy"),

        q2_glove=np.load("Arrays2/embedding_matrix.npy"),
        q2_fasttext=np.load("Arrays2/embedding_matrix_fasttext.npy"),
        # q2_lda = np.load("Arrays2/embedding_matrix_lda.npy")
    )

    return answers, embeddings


KFOLDS = 5


def get_train_sequences(n, features, labels):
    """Get training and validation sequences based on kfolds cross validation"""
    y, _ = score(labels, labels)

    kf = StratifiedKFold(KFOLDS, True, 1)
    split = list(kf.split(features, y))[n]

    x_train = np.array(features[split[0]])
    x_valid = np.array(features[split[1]])

    y_train = np.array(labels[split[0]])
    y_valid = np.array(labels[split[1]])

    return x_train, x_valid, y_train, y_valid


def get_parametergrid():
    # Hyperparameters
    hyperparameters = {
        "1_question": ["1", "2"],
        "2_train": ["freeze", "train"],
        "3_rnn": ["lstm", "gru", "baseline"],
        "4_bi": ["", "bi"],
        "5_att": ["", "att"],
        "6_emb": ["glove", "fasttext", "lda"],
    }

    return ParameterGrid(hyperparameters)


def get_nonbaseline_grid():
    """Gridsearch for all non-baseline models"""
    grid = []

    for p in get_parametergrid():
        # Don't train lda for question 2 cos we don't actually have the lda embeddings yet
        if p["6_emb"] == "lda":
            continue

        # Don't train baseline models, that's a separate function
        if p["3_rnn"] == "baseline":
            continue

        # Filter out models we don't intend to train
        if p["5_att"] == "att" and p["4_bi"] != "bi":
            continue
        elif p["5_att"] == "att":
            pass
        elif p["5_att"] == "" and p["6_emb"] != "glove":
            continue

        p["filename"] = "Models/q%s/%s/%s/%s%s%s/" % (p["1_question"], p["2_train"], p["3_rnn"],
                                                      p["4_bi"], p["6_emb"], p["5_att"])
        grid.append(p)

    return grid


def get_baseline_grid():
    """Gridsearch for baseline models"""
    grid = []

    for p in get_parametergrid():
        if p["3_rnn"] != "baseline" or p["4_bi"] == "bi" or p["5_att"] == "att" or p["6_emb"] != "glove":
            continue

        p["filename"] = "Models/q%s/%s/%s/" % (p["1_question"], p["2_train"], p["3_rnn"])
        grid.append(p)

    return grid


CURR = get_baseline_grid() + get_nonbaseline_grid()


# Find out which model number a particular filename is based on the current model name
def test(x):
    saved = 0
    for i in CURR:
        filename = "Models/q%s/%s/%s/%s%s%s/" % (
            i["1_question"], i["2_train"], i["3_rnn"], i["4_bi"], i["6_emb"], i["5_att"])

        for k in range(KFOLDS):
            f = filename + " " + str(k)
            if f == x:
                return saved
            else:
                saved += 1


# Find out how many models have been trained -- total should be 200
def num_models(x):
    count = 0
    for i in x:
        count += len(os.listdir(i["filename"]))
    return count

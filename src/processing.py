import pandas as pd
import numpy as np
import io
import json
import pickle
import nltk
import string
from pathlib import Path

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from initialisation import load_arrays, score
import plotly.offline as py
import plotly.graph_objs as go

FILEPATH = Path(__file__).parent.absolute()


def get_training_data():
    """Load the training data from CSV file and remove entries where there is no student answer"""
    data = pd.read_excel(FILEPATH/"data"/"thermal_physics_quiz_dataset.xlsx", sheet_name=None)
    t = list(data.keys())
    training_data = pd.concat([data[t[1]],
                               data[t[2]].drop(['Unnamed: 4'], axis=1),
                               data[t[3]].drop(data[t[3]].columns[[2, 3, 6]], axis=1)])

    training_data.index = range(len(training_data))
    training_data.drop(training_data[training_data.iloc[:, 2] == " "].index, inplace=True)

    return training_data


def clean_tokenize(corpus):
    """Split essay into tokens, removing punctuations and non-alphabetical tokens and also removing stop words"""
    # Tokenize the answers
    corpus = nltk.word_tokenize(corpus)

    # Make all the words lowercase
    corpus = [w.lower() for w in corpus]

    # Remove punctuation
    punctuation = "".join([i for i in string.punctuation if i != "/"])
    punc_table = str.maketrans('/', ' ', punctuation)
    corpus = [w.translate(punc_table) for w in corpus]

    # Remove non-alphabetical tokens
    corpus = [w for w in corpus if w.isalpha()]

    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    corpus = [w for w in corpus if w not in stop_words]

    return corpus


def make_sequences(texts):
    """Turn each word of an essay into an integer based on a common dictionary"""

    # Create the tokenizer object and train on texts
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts

    print(f'There are {num_words} unique words.')

    # Convert text to sequences of integers
    sequences = np.array(tokenizer.texts_to_sequences(texts))

    sequences = pad_sequences(sequences)

    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, sequences


def load_glove(word_idx, num_words):
    """Load GloVe embeddings (300d)
    GloVe embeddings not in this folder because it's like 1GB so go download it from
    https://nlp.stanford.edu/projects/glove/
    """

    glove = np.loadtxt(FILEPATH/"data"/"glove.6B.300d.txt", dtype='str', comments=None)

    word_lookup = {}
    count = 0

    for line in glove:
        if line[0] in word_idx.keys():
            count += 1
            word_lookup[line[0]] = line[1:].astype('float')

        if count == len(word_idx.keys()):
            break

    # New matrix to hold word embeddings
    embedding_matrix = np.zeros((num_words, 300))

    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector

    return embedding_matrix, word_lookup


def load_fasttext(word_idx, num_words):
    """Load FastText embeddings (300d).
    Same as for GloVe, download the embeddings from https://fasttext.cc
    """
    fin = io.open(FILEPATH/"data"/"crawl-300d-2M-subword.vec", 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    word_lookup = {}

    count = 0

    for line in fin:
        tokens = line.rstrip().split(' ')

        if tokens[0] in word_idx.keys():
            count += 1
            word_lookup[tokens[0]] = np.array(list(map(float, tokens[1:])))

        if count == len(word_idx.keys()):
            break

    # New matrix to hold word embeddings
    embedding_matrix = np.zeros((num_words, d))

    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector

    return n, d, embedding_matrix, word_lookup


def load_lda(word_lookup, word_idx):
    """Load LDA word embeddings generated from Advay's code"""
    # New matrix to hold word embeddings
    embedding_matrix = np.zeros((len(word_idx) + 1, 300))

    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
    return embedding_matrix


def load_doc_embeddings(doc_embeddings, sequences):
    """Load LDA document embeddings generated from Advay's code"""
    output = []

    for i, x in enumerate(sequences):
        doc_em = []
        for word in x:
            if word == 0:
                doc_em.append(np.zeros(300))
            else:
                doc_em.append(doc_embeddings[str(i)])
        output.append(np.array(doc_em))

    return np.array(output)


def generate_arrays(qn):
    """Main function to generate the arrays. Refer to README.md for a list and description of the arrays generated
    """
    training_data = get_training_data()
    if qn == 1:
        answers = np.array([clean_tokenize(i) for i in training_data.iloc[:, 0]])
        scores = np.array(to_categorical(training_data.iloc[:, 1]))
    else:
        answers = np.array([clean_tokenize(i) for i in training_data.iloc[:, 2]])
        scores = np.array(to_categorical(training_data.iloc[:, 3]))

    word_idx, idx_word, num_words, word_counts, sequences = make_sequences(answers)

    print("Loading GloVe...")
    embedding_matrix_glove, word_lookup_glove = load_glove(word_idx, num_words)

    print("Loading fastText...")
    n, d, embedding_matrix_fasttext, word_lookup_fasttext = load_fasttext(word_idx, num_words)

    path = FILEPATH/"arrays/q"

    if qn == 1:
        print("Loading LDA...")  # Only for Q1
        doc_embeddings = json.load(open(f"{path}1/Lda2Vec/q1_embeddings/q1_doc_embeddings.json"))
        word_embeddings = json.load(open(f"{path}1/Lda2Vec/q1_embeddings/q1_word_embeddings.json"))
        embedding_matrix_lda = load_lda(word_embeddings, word_idx)
        doc_embedding_inputs = load_doc_embeddings(doc_embeddings, sequences)
        np.save(f"{path}{qn}/embedding_matrix_lda.npy", embedding_matrix_lda)
        np.save(f"{path}{qn}/doc_embedding_inputs.npy", doc_embedding_inputs)

    # Save arrays
    np.save(f"{path}{qn}/sequences.npy", sequences)
    np.save(f"{path}{qn}/scores.npy", scores)
    pickle.dump(word_idx, open(f"{path}{qn}/word_idx.pickle", "wb"))
    pickle.dump(idx_word, open(f"{path}{qn}/idx_word.pickle", "wb"))

    np.save(f"{path}{qn}/embedding_matrix_glove.npy", embedding_matrix_glove)
    np.save(f"{path}{qn}/embedding_matrix_fasttext.npy", embedding_matrix_fasttext)


def score_distribution(save=False):
    """Plot the scores of responses across both datasets/questions"""
    answers, _ = load_arrays()
    scores1 = answers["q1_scores"]
    scores2 = answers["q2_scores"]

    scores1 = score(scores1)
    scores2 = score(scores2)

    layout = go.Layout(  # title=dict(text="Scores across Both Datasets", x=0.5, xanchor="center"),
        xaxis=dict(title="Score", nticks=5),
        yaxis=dict(title="Number of Student Answers"),
        bargap=0.5,
        margin=dict(t=0, b=0, l=0, r=0))

    fig = go.Figure(data=[go.Histogram(name="Dataset 1", x=scores1[:]),
                          go.Histogram(name="Dataset 2", x=scores2[:])],
                    layout=layout)

    for i in range(3):
        y = np.unique(scores1[:], return_counts=True)[1][i]

        fig.add_annotation(
            go.layout.Annotation(
                x=i - 0.125,
                y=y / 2,
                text=str(y),
                showarrow=False,
            )
        )

    for i in range(4):
        y = np.unique(scores2[:], return_counts=True)[1][i]

        fig.add_annotation(
            go.layout.Annotation(
                x=i + 0.125,
                y=y / 2,
                text=str(y),
                showarrow=False,
            )
        )

    py.iplot(fig, filename='basic histogram')

    if save:
        fig.write_image("combined_scores.pdf", format="pdf")


def word_distribution(save=False):
    """Plot the number of words for responses across both datasets/questions"""
    answers, _ = load_arrays()
    answers1 = answers["q1_answers"]
    answers2 = answers["q2_answers"]

    layout = go.Layout(  # title=dict(text="Number of Words across Both Datasets", x=0.5, xanchor="center"),
        xaxis=dict(title="Length of Responses", nticks=4),
        yaxis=dict(title="Number of Student Responses"),
        margin=dict(t=0, b=0, l=0, r=0))

    x1 = [len(i) for i in answers1]
    x2 = [len(i) for i in answers2]

    fig = go.Figure(data=[go.Histogram(name="Dataset 1", x=x1),
                          go.Histogram(name="Dataset 2", x=x2)],
                    layout=layout)

    x = 150
    y = 37
    y2 = y - 3

    fig.add_annotation(
        go.layout.Annotation(
            x=x,
            y=y,
            text="Mean Length of Responses for Dataset 1: %d" % (sum(x1) / len(x1)),
            showarrow=False,
        )
    )

    fig.add_annotation(
        go.layout.Annotation(
            x=x,
            y=y2,
            text="Mean Length of Responses for Dataset 2: %d" % (sum(x2) / len(x2)),
            showarrow=False,
        )
    )

    py.iplot(fig, filename='basic histogram')

    if save:
        fig.write_image("combined_words.pdf", format="pdf")


if __name__ == '__main__':
    print("Generating arrays for question 1...")
    generate_arrays(1)

    print("Generating arrays for question 2")
    generate_arrays(2)

    print("Generating score distribution...")
    score_distribution()

    print("Generating word distribution...")
    word_distribution()

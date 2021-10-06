from src import processing
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_marking_scheme():
    with open("q1_marking_scheme.txt", "r") as file1, \
            open("../src/Arrays/word_idx.pickle", "rb") as file2:
        marking_scheme = [line for line in file1]
        marking_scheme = " ".join(marking_scheme)
        marking_scheme = processing.clean_tokenize(marking_scheme)

        word_idx = pickle.load(file2)

        marking_scheme = [word_idx.get(word, word) for word in marking_scheme]

        marking_scheme = pad_sequences([marking_scheme], maxlen=152)

    return marking_scheme


if __name__ == '__main__':
    get_marking_scheme()

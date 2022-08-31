import nltk
from pathlib import Path
import requests
import zipfile

FILEPATH = Path(__file__).parent.absolute()


def download(url, zippath):
    r = requests.get(url, stream=True)

    with open(zippath, "wb") as file:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)


def extract(zippath, embeddings_path):
    with zipfile.ZipFile(zippath, 'r') as zip_ref:
        zip_ref.extractall(embeddings_path)


def download_glove():
    glove_url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    glove_zip = FILEPATH/'src'/'data'/'glove.6B.zip'
    glove_folder = FILEPATH/'src'/'data'/'glove.6B'
    glove_file = FILEPATH/'src'/'data'/'glove.6B.300d.txt'

    if not glove_file.exists():  # if the file does not exist
        if not glove_folder.exists():
            if not glove_zip.exists():  # if zip file is not downloaded
                print("Downloading GloVe...")
                download(glove_url, glove_zip)
            print("GloVe downloaded!")
            print("Extracting GloVe...")
            extract(glove_zip, glove_folder)
        (glove_folder/'glove.6B.300d.txt').rename(glove_file)
    print("GloVe extracted!")

    if glove_folder.exists():
        for child in glove_folder.glob("**/*"):
            child.unlink()
        glove_folder.rmdir()

    if glove_zip.exists():
        glove_zip.unlink()


def download_fasttext():
    fasttext_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
    fasttext_zip = FILEPATH/'src'/'data'/'crawl-300d-2M.vec.zip'
    fasttext_folder = FILEPATH/'src'/'data'/'crawl-300d-2M'
    fasttext_file = FILEPATH/'src'/'data'/'crawl-300d-2M.vec'

    if not fasttext_file.exists():
        if not fasttext_folder.exists():
            if not fasttext_zip.exists():
                print("Downloading fastText...")
                download(fasttext_url, fasttext_zip)
            print("fastText downloaded!")
            print("Extracting fastText...")
            extract(fasttext_zip, fasttext_folder)
        (fasttext_folder/'crawl-300d-2M.vec').rename(fasttext_file)
    print("fastText extracted!")

    if fasttext_folder.exists():
        fasttext_folder.rmdir()

    if fasttext_zip.exists():
        fasttext_zip.unlink()


def setup():
    nltk.download('punkt')
    nltk.download('stopwords')
    download_glove()
    download_fasttext()


if __name__ == '__main__':
    setup()

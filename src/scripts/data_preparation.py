import json
import zipfile
import glob
from typing import List, Tuple

import logging
import re

import gensim
import pymongo as pymongo
import src.scripts.config as config
from gensim.models.phrases import Phraser
from pymongo.collection import Collection
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer

discard_pattern = re.compile("^[,.:;\-_\"\'“”—()\[\]|/!?–]+$")


def split_and_tokenize(document: str) -> List[List[str]]:
    result = []
    sentences = split_multi(document)
    for sentence in sentences:
        result.append([token.lower() for token in web_tokenizer(sentence) if not discard_pattern.match(token)])
    return result


class WebhoseZipFileProcessor(object):

    @staticmethod
    def _read_webhose_zip_files(directory_name: str, output_file: str) -> None:
        file_names = glob.glob(directory_name + "/*.zip")
        print("Listed {} files in directory {}".format(len(file_names), directory_name))
        for file_name in file_names:
            print("File: {}".format(file_name))
        for file_name in file_names:
            WebhoseZipFileProcessor._read_webhose_zip_file(file_name, output_file)

    @staticmethod
    def _read_webhose_zip_file(file_name: str, output_file: str) -> None:
        with zipfile.ZipFile(file_name, "r") as zip:
            with open(output_file, "a") as out:
                num_docs_read = 0
                for file in zip.namelist():
                    with zip.open(file) as json_file:
                        data = json_file.read()
                        j = json.loads(data.decode("utf-8"))
                        try:
                            title = j["title"]
                            text = j["text"]
                        except KeyError:
                            continue
                        num_docs_read += 1
                        if num_docs_read % 1000 == 0:
                            print("Read {} documents".format(num_docs_read))
                        for sentence in split_and_tokenize(title + ". " + text):
                            if len(sentence) > 4:
                                out.write(" ".join(sentence) + "\n")

    @staticmethod
    def process(input_directory_name: str, output_file: str) -> None:
        WebhoseZipFileProcessor._read_webhose_zip_files(input_directory_name, output_file)


class DbProcessor(object):

    def __init__(self, db_url: str, db_name: str, collection_name: str):
        self.db_url = db_url
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = self._set_up_db(db_url, db_name, collection_name)

    @staticmethod
    def _set_up_db(db_url: str, db_name: str, collection_name: str) -> Collection:
        client = pymongo.MongoClient(db_url)
        return client[db_name][collection_name]

    def _read_db(self, num_docs=None) -> List[List[str]]:
        if num_docs:
            cursor = self.collection.find().limit(num_docs)
        else:
            cursor = self.collection.find()
        num_docs_read = 0
        for document in cursor:
            try:
                title = document["objects"][0]["title"]
                text = document["objects"][0]["text"]
            except KeyError:
                continue
            num_docs_read += 1
            if num_docs_read % 500 == 0:
                print("Read {} documents".format(num_docs_read))
            for sentence in split_and_tokenize(title + ". " + text):
                yield sentence

    def process(self, output_file: str, num_docs=None) -> None:
        with open(output_file, "w") as out:
            for sentence in self._read_db(num_docs=num_docs):
                if len(sentence) > 4:
                    out.write(" ".join(sentence) + "\n")


def create_phrases(input_directory: str, bigram_model_file: str, trigram_model_file: str) -> None:
    sentences = gensim.models.word2vec.PathLineSentences(input_directory)
    bigrams = gensim.models.phrases.Phrases(sentences, min_count=10)
    bigrams_phraser = gensim.models.phrases.Phraser(bigrams)
    bigrams_phraser.save(bigram_model_file)
    trigrams = gensim.models.phrases.Phrases(bigrams[sentences], min_count=10)
    trigrams_phraser = gensim.models.phrases.Phraser(trigrams)
    trigrams_phraser.save(trigram_model_file)


def load_phrase_models(bigram_model_file: str, trigram_model_file: str) -> Tuple[Phraser, Phraser]:
    bigram_model = gensim.models.phrases.Phraser.load(bigram_model_file)
    trigram_model = gensim.models.phrases.Phraser.load(trigram_model_file)
    return bigram_model, trigram_model


def analyze_file(input_text_file: str, output_text_file: str, bigram_model_file: str, trigram_model_file: str) -> None:
    print("Reading raw text from {}".format(input_text_file))
    print("Writing phrases to {}".format(output_text_file))
    bigram_model, trigram_model = load_phrase_models(bigram_model_file, trigram_model_file)
    num_lines_written = 0
    with open(output_text_file, "w") as output:
        with open(input_text_file, "r") as input:
            for line in input:
                if num_lines_written % 10000 == 0:
                    print("{} lines written".format(num_lines_written))
                output.write(" ".join(list(trigram_model[bigram_model[line.rstrip().split()]])) + "\n")
                num_lines_written += 1


if __name__ == "__main__":
    """
    The following sequence of method calls reads a MongoDb collection containing texts extracted from approx. 6000
    articles available in the MetaCurate database, as well as English corpora from webhose.io. The texts are segmented
    into sentences and tokenized, then used for training Gensim Phrasers, identifying collocations, for bigrams and
    tri- and quadgrams. The Phrasers are save disk as well as used to process and print a new version of the initial
    raw text. 
    """
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)
    base_directory = "/Users/fredriko/Dropbox/data/metacurate-lexicon/"

    # Access to the local MongoDb containing MetaCurate texts.
    db_url = "mongodb://localhost:27017"
    db_name = "texts"
    db_collection_name = "diffbot"

    # The directory in which the first level of unzipped webhose.io corpora are available. The corpora are retrieved
    # from https://webhose.io/datasets/, e.g., "English news articles", and "Technology news articles".
    webhose_zip_directory = "/Users/fredriko/Dropbox/data/metacurate-lexicon/zip/webhose-unzipped"

    # Specification of where to put extracted and processed texts.
    raw_db_text = config.RAW_DATA_DIRECTORY + "metacurate-out.txt"
    raw_webhose_text = config.RAW_DATA_DIRECTORY + "webhose-out.txt"

    # Specification of where to put the phrase data
    phrases_db_text = config.PHRASE_DATA_DIRECTORY + "metacurate-phrases.txt"
    phrases_webhose_text = config.PHRASE_DATA_DIRECTORY + "webhose-phrases.txt"

    # Specification of where to store Phraser models
    bigram_model_file = config.PHRASE_MODELS_DIRECTORY + "bigram_phrases.model"
    trigram_model_file = config.PHRASE_MODELS_DIRECTORY + "trigram_phrases.model"

    db_processor = DbProcessor(db_url, db_name, db_collection_name)
    db_processor.process(raw_db_text, num_docs=10000)

    webhose_processor = WebhoseZipFileProcessor()
    webhose_processor.process(config.WEBHOSE_ZIP_DIRECTORY, raw_webhose_text)

    # 2018-06-30 22:32:18,683: INFO: collected 38031498 word types from a corpus of 440206564 words (unigram + bigrams) and 20817495 sentences
    create_phrases(config.RAW_DATA_DIRECTORY, bigram_model_file, trigram_model_file)
    
    analyze_file(raw_db_text, phrases_db_text, bigram_model_file, trigram_model_file)
    analyze_file(raw_webhose_text, phrases_webhose_text, bigram_model_file, trigram_model_file)

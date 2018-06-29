import json
import zipfile
import glob
from typing import List

import logging
import re
import pymongo as pymongo
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
            # [list(gen()) for gen in generator_of_generator_functions]
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



"""
def create_phrases():
    db_host = "mongodb://localhost:27017"
    db_name = "texts"
    db_collection_name = "diffbot"
    common_terms = ["of", "with", "without", "and", "or", "the", "a", "an"]
    sentences = read_db(db_host, db_name, db_collection_name)
    bigrams = gensim.models.phrases.Phrases(sentences, common_terms=common_terms)
    bigrams.save("bigrams_model")
    trigrams = gensim.models.phrases.Phrases(bigrams[sentences], common_terms=common_terms)
    trigrams.save("trigrams_model")
    quadgrams = gensim.models.phrases.Phrases(trigrams[bigrams[sentences]], common_terms=common_terms)
    quadgrams.save("quadgrams_model")
"""


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)

    db_url = "mongodb://localhost:27017"
    db_name = "texts"
    db_collection_name = "diffbot"
    db_output_file = "metacurate-out.txt"

    webhose_zip_directory = "/Users/fredriko/data/webhose-corpora/unzipped"
    webhose_output_file = "webhose-out.txt"

    db_processor = DbProcessor(db_url, db_name, db_collection_name)
    db_processor.process(db_output_file, num_docs=10000)

    webhose_processor = WebhoseZipFileProcessor()
    webhose_processor.process(webhose_zip_directory, webhose_output_file)

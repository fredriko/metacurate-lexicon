import logging
import os
from typing import Union

import gensim
import src.scripts.config as config


def train_fasttext_model(input: str, output_directory: str, model_name: str) -> None:

    if not os.access(output_directory, os.W_OK):
        print("Cannot write to directory {}. Exiting!".format(output_directory))
        exit(1)

    if os.path.isdir(input):
        sentences = gensim.models.word2vec.PathLineSentences(input)
    else:
        sentences = gensim.models.word2vec.LineSentence(input)

    #model = gensim.models.FastText(sentences, sg=0, size=150, window=10, min_count=4, workers=10, iter=4)
    model = gensim.models.FastText(workers=10, window=10, min_count=10)
    model.build_vocab(sentences)
    #model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    model.save(output_directory + model_name)


def train_word2vec_model(input: str, output_directory: str, model_name: str) -> None:

    if not os.access(output_directory, os.W_OK):
        print("Cannot write to directory {}. Exiting!".format(output_directory))
        exit(1)

    if os.path.isdir(input):
        sentences = gensim.models.word2vec.PathLineSentences(input)
    else:
        sentences = gensim.models.word2vec.LineSentence(input)

    model = gensim.models.Word2Vec(sentences, sg=0, size=150, window=10, min_count=1, workers=10)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    model.save(output_directory + model_name)


def print_result(model: Union[gensim.models.FastText, gensim.models.Word2Vec], word: str, top_n: int) -> None:
    results = model.wv.most_similar(positive=word.replace(" ", "_"), topn=top_n)
    print("Most similar to '{}'".format(word))
    for result in results:
        text = str(result[0])
        score = result[1]
        text = text.replace("_", " ")
        print(text, score)
    print(" ")


def ocular_inspection(model_file: str, top_n: int = 10) -> None:
    model = gensim.models.Word2Vec.load(model_file)
    print_result(model, "artificial intelligence", top_n)
    print_result(model, "google", top_n)
    print_result(model, "language", top_n)
    print_result(model, "self-driving cars", top_n)
    print_result(model, "autonomous vehicles", top_n)
    print_result(model, "gdpr", top_n)
    print_result(model, "drones", top_n)
    print_result(model, "donald trump", top_n)
    print_result(model, "amazon", top_n)
    print_result(model, "white house", top_n)
    print_result(model, "silicon valley", top_n)
    print_result(model, "apple", top_n)
    print_result(model, "suit", top_n)
    print_result(model, "jeans", top_n)
    print_result(model, "san francisco", top_n)
    print_result(model, "one of the most popular", top_n)
    print_result(model, "excel files", top_n)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)
    train_fasttext_model(config.PHRASE_DATA_DIRECTORY + "metacurate-phrases.txt", config.WORDSPACE_MODELS_DIRECTORY, "fasttext-metacurate-cbow-1.model")
    #train_word2vec_model(config.PHRASE_DATA_DIRECTORY + "metacurate-phrases.txt", config.WORDSPACE_MODELS_DIRECTORY, "word2vec-metacurate-cbow.model")
    ocular_inspection(config.WORDSPACE_MODELS_DIRECTORY + "fasttext-metacurate-cbow-1.model")
    #ocular_inspection(config.WORDSPACE_MODELS_DIRECTORY + "word2vec-metacurate-cbow.model")

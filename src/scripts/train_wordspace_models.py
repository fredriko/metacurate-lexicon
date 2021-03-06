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

    model = gensim.models.FastText(workers=10, window=10, min_count=20, size=100)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    model.save(output_directory + model_name)
    # We want the vectors only to reduce memory footprint: this is the file(s) that the oneline lexicon should use.
    vectors = model.wv
    vectors.save(output_directory + model_name + ".vectors-only")


def train_word2vec_model(input: str, output_directory: str, model_name: str) -> None:

    if not os.access(output_directory, os.W_OK):
        print("Cannot write to directory {}. Exiting!".format(output_directory))
        exit(1)

    if os.path.isdir(input):
        sentences = gensim.models.word2vec.PathLineSentences(input)
    else:
        sentences = gensim.models.word2vec.LineSentence(input)

    model = gensim.models.Word2Vec(sentences, sg=0, size=100, window=10, min_count=20, workers=10)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    model.save(output_directory + model_name)
    # We want the vectors only to reduce memory footprint: this is the file(s) that the oneline lexicon should use.
    vectors = model.wv
    vectors.save(output_directory + model_name + ".vectors-only")


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
    print("Size of lexicon: {} terms".format(len(model.wv.vocab)))
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

# 2018-07-07 21:12:24,995: INFO: EPOCH - 10 : training on 195069967 raw words (151372237 effective words) took 213.2s, 710002 effective words/s
if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)
    #train_fasttext_model(config.PHRASE_DATA_DIRECTORY + "metacurate-phrases.txt", config.WORDSPACE_MODELS_DIRECTORY, "fasttext-metacurate-cbow-2.model")
    train_word2vec_model(config.SPLITS_DATA_DIRECTORY_10M, config.WORDSPACE_MODELS_DIRECTORY, "word2vec-metacurate-cbow-20M-100-w10-min20-split.model")
    #ocular_inspection(config.WORDSPACE_MODELS_DIRECTORY + "fasttext-metacurate-cbow-2.model")
    ocular_inspection(config.WORDSPACE_MODELS_DIRECTORY + "word2vec-metacurate-cbow-20M-100-w10-min20-split.model")

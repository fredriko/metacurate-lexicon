import logging

import gensim

def train(input_directory: str) -> None:
    sentences = gensim.models.word2vec.PathLineSentences(input_directory)
    print("Setting up mopdel")
    #model = gensim.models.Word2Vec(sentences, sg=1, size=150, window=10, min_count=4, workers=10)

    model = gensim.models.FastText(sentences, sg=0, size=150, window=10, min_count=4, workers=10)
    model.build_vocab(sentences)
    print("Training model")
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    print("Saving model")
    model.save("fasttext-cbow.model")


def print_result(model, word, top_n):
    results = model.wv.most_similar(positive=word.replace(" ", "_"), topn=top_n)
    print("Most similar to '{}'".format(word))
    for result in results:
        text = str(result[0])
        score = result[1]
        text = text.replace("_", " ")
        print(text, score)
    print(" ")


def test():
    topn = 10
    model = gensim.models.Word2Vec.load("test_model_1")
    #    for v in model.wv.vocab:
    #        if "_" in v:
    #            print(v)
    print_result(model, "artificial intelligence", topn)
    print_result(model, "google", topn)
    print_result(model, "language", topn)
    print_result(model, "self-driving cars", topn)
    print_result(model, "autonomous vehicles", topn)
    print_result(model, "gdpr", topn)
    print_result(model, "drones", topn)
    print_result(model, "donald trump", topn)
    print_result(model, "amazon", topn)
    print_result(model, "white house", topn)
    print_result(model, "silicon valley", topn)
    print_result(model, "apple", topn)
    print_result(model, "suit", topn)
    print_result(model, "jeans", topn)
    print_result(model, "san francisco", topn)
    print_result(model, "one of the most popular", topn)
    print_result(model, "excel files", topn)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)
    train("/Users/fredriko/Dropbox/data/wordspaces/phrases")
    test()

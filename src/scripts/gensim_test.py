import logging

import gensim

def load_and_test():
    bigrams = gensim.models.phrases.Phrases.load("bigrams_model")
    trigrams = gensim.models.phrases.Phrases.load("trigrams_model")
    quadgrams = gensim.models.phrases.Phrases.load("quadgrams_model")
    db_host = "mongodb://localhost:27017"
    db_name = "texts"
    db_collection_name = "diffbot"
    sentences = read_db(db_host, db_name, db_collection_name, num_docs=3)
    for sentence in list(quadgrams[trigrams[bigrams[sentences]]]):
        print(sentence)


def train():
    db_host = "mongodb://localhost:27017"
    db_name = "texts"
    db_collection_name = "diffbot"

    bigrams = gensim.models.phrases.Phrases.load("bigrams_model")
    trigrams = gensim.models.phrases.Phrases.load("trigrams_model")
    quadgrams = gensim.models.phrases.Phrases.load("quadgrams_model")

    sentences = read_db(db_host, db_name, db_collection_name)
    n_gram_sentences = list(quadgrams[trigrams[bigrams[sentences]]])
    print("Setting up mopdel")
    model = gensim.models.Word2Vec(n_gram_sentences, size=150, window=10, min_count=2, workers=10)
    print("Training model")
    model.train(n_gram_sentences, total_examples=len(n_gram_sentences), epochs=10)
    print("Saving model")
    model.save("test_model_1")


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
    load_and_test()
    train()
    test()

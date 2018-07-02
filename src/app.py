import gensim
from flask import Flask, render_template
from typing import Union

import src.scripts.config as config

app = Flask(__name__)


def print_result(model: Union[gensim.models.FastText, gensim.models.Word2Vec], word: str, top_n: int) -> None:
    results = model.wv.most_similar(positive=word.replace(" ", "_"), topn=top_n)
    print("Most similar to '{}'".format(word))
    fixed = []
    fixed.append("<ol>\n")

    for result in results:
        text = str(result[0])
        score = result[1]
        text = text.replace("_", " ")
        print(text, score)
        fixed.append("<li>" + text + ": " + str(score))
    print(" ")
    fixed.append("</ol>\n")
    return " ".join(fixed)


MODEL = gensim.models.Word2Vec.load(config.WORDSPACE_MODELS_DIRECTORY + "fasttext-metacurate-cbow.model")


@app.route("/")
def index():
    #model = gensim.models.Word2Vec.load(config.WORDSPACE_MODELS_DIRECTORY + "fasttext-metacurate-cbow.model")
    #print_result(model)
    return "<a href=\"/lookup/gensim\">lookup!</a>"


@app.route("/lookup/<string:term>")
def lookup(term=None):
    if term is not None:
        results =   print_result(MODEL, term, 10)
        return f"seeing {term}" + results

    else:
        return "Please supply term!"

import platform
from flask import Flask, request, render_template, g
from gensim.models import KeyedVectors, phrases

from src.scripts import config
from src.models.api.views import api_blueprint

app = Flask(__name__)


heroku_model_root = "/app/gensim-models/"

def load_vectors():
    vectors_file_name = "word2vec-metacurate-cbow-5M-100-w10-min20-split.vectors"
    print(f"Node:{platform.node()}")
    if platform.system() == "Darwin" or platform.node() == "vector":
        # I'm on a Mac or I'm on a magic machine.
        vectors_path = config.WORDSPACE_MODELS_DIRECTORY + vectors_file_name
    else:
        # Here's where heroku looks for the vectors.
        vectors_path = heroku_model_root + vectors_file_name
    return KeyedVectors.load(vectors_path)


def load_phrasers():
    bigram_model_name = "bigram_phrases.model"
    trigram_model_name = "trigram_phrases.model"
    if platform.system() == "Darwin" or platform.node() == "vector":
        bigram_path = config.PHRASE_MODELS_DIRECTORY + bigram_model_name
        trigram_path = config.PHRASE_MODELS_DIRECTORY + trigram_model_name
    else:
        bigram_path = heroku_model_root + bigram_model_name
        trigram_path = heroku_model_root + trigram_model_name
    bigrams = phrases.Phrases.load(bigram_path)
    trigrams = phrases.Phraser.load(trigram_path)
    return bigrams, trigrams


MODEL = load_vectors()
BIGRAMS, TRIGRAMS = load_phrasers()


@app.before_request
def before_request():
    g.metacurate_vectors = MODEL
    g.metacurate_bigrams = BIGRAMS
    g.metacurate_trigrams = TRIGRAMS


@app.route("/lookup/")
def lookup():
    term = request.args.get("term")
    result = []
    similarities = []
    error = None
    if term is not None:
        term = term.lower().strip()
        if len(term) > 0:
            try:
                similarities = g.metacurate_vectors.most_similar(positive=term.replace(" ", "_"), topn=10)
            except KeyError:
                error = {"term": term, "message": "The term is not in the lexicon"}
            for similarity in similarities:
                text = str(similarity[0]).replace("_", " ")
                score = str(round(similarity[1], 2))
                result.append({"term": text, "score": score})
        else:
            error = {"term": None, "message": "No term specified!"}
    return render_template("home.jinja2", data={"lookup": term, "similarities": result}, error=error)


@app.route("/")
def index():
    return render_template("home.jinja2", data=None, error=None)


@app.route("/about")
def about():
    return render_template("about.jinja2")


app.register_blueprint(api_blueprint)

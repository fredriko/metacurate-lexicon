import platform
from flask import Flask, request, render_template, g
from gensim.models import KeyedVectors

from src.scripts import config
from src.models.api.views import api_blueprint

app = Flask(__name__)


def load_vectors():
    vectors_file_name = "word2vec-metacurate-cbow-5M-100-w10-min20-split.vectors"
    if platform.system() == "Darwin":
        # I'm on a Mac.
        vectors_path = config.WORDSPACE_MODELS_DIRECTORY + vectors_file_name
    else:
        # Here's where heroku looks for the vectors.
        vectors_path = "/app/gensim-models/" + vectors_file_name
    return KeyedVectors.load(vectors_path)


MODEL = load_vectors()


@app.before_request
def before_request():
    g.metacurate_vectors = MODEL


@app.route("/", methods=["GET", "POST"])
def lookup():
    result = []
    similarities = []
    term = None
    error = None
    if request.method == "POST":
        term: str = request.form["term"]
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


@app.route("/about")
def about():
    return render_template("about.jinja2")


app.register_blueprint(api_blueprint)
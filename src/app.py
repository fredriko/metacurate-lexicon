import gensim
import platform
from flask import Flask, request, render_template, g

from src.scripts import config
from src.models.api.views import api_blueprint

app = Flask(__name__)


def load_model():
    model_file_name = "word2vec-metacurate-cbow-5M-100-w10-min20-split.model"
    if platform.system() == "Darwin":
        # I'm on a Mac.
        model_path = config.WORDSPACE_MODELS_DIRECTORY + model_file_name
    else:
        # Here's where heroku looks for the model.
        model_path = "/app/gensim-models/" + model_file_name
    return gensim.models.Word2Vec.load(model_path)


MODEL = load_model()


@app.before_request
def before_request():
    g.metacurate_model = MODEL


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
                similarities = g.metacurate_model.wv.most_similar(positive=term.replace(" ", "_"), topn=10)
            except KeyError:
                error = {"term": term, "message": "The term is not in the lexicon"}
            for similarity in similarities:
                text = str(similarity[0]).replace("_", " ")
                score = str(round(similarity[1], 2))
                result.append({"term": text, "score": score})
        else:
            error = {"term": None, "message": "No term specified!"}
    return render_template("home.jinja2", data={"lookup": term, "similarities": result}, error=error)


app.register_blueprint(api_blueprint)
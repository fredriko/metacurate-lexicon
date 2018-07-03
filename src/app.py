import gensim
from flask import Flask, request, render_template

app = Flask(__name__)


#MODEL = gensim.models.Word2Vec.load(config.WORDSPACE_MODELS_DIRECTORY + "fasttext-metacurate-cbow.model")
MODEL = gensim.models.Word2Vec.load("fasttext-metacurate-cbow.model")

@app.route("/")
def index():
    return render_template("home.jinja2")


@app.route("/lookup", methods=["GET", "POST"])
def lookup():
    result = []
    term = None
    error = None
    if request.method == "POST":
        term: str = request.form["term"]
        term = term.lower().strip()
        if len(term) > 0:
            similarities = MODEL.wv.most_similar(positive=term.replace(" ", "_"), topn=10)
            for similarity in similarities:
                text = str(similarity[0]).replace("_", " ")
                score = str(similarity[1])
                result.append({"term": text, "score": score})
        else:
            error = "No term specified!"
    return render_template("home.jinja2", data={"lookup": term, "similarities": result}, error=error)


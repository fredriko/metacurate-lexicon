from flask import Blueprint, render_template, request, jsonify, session, url_for, g

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/lookup", defaults={"term": None})
@api_blueprint.route("/lookup/<string:term>", defaults={"num": 10})
@api_blueprint.route("/lookup/<string:term>/<int:num>")
def lookup_term(term, num):
    result = []
    similarities = []
    error = None
    if term:
        term = term.lower().strip()
        if len(term) > 0:
            try:
                similarities = g.metacurate_model.wv.most_similar(positive=term.replace(" ", "_"), topn=num)
            except KeyError:
                error = {"term": term, "error": "The term is not in the lexicon"}
            for similarity in similarities:
                text = str(similarity[0]).replace("_", " ")
                score = str(round(similarity[1], 2))
                result.append({"term": text, "score": score})
        else:
            error = {"term": None, "error": "No term specified!"}
    else:
        error = {"term": None, "error": "No term specified!"}
    if error:
        result = error
    return jsonify(result)

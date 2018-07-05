from flask import Blueprint, g
from flask_restplus import Resource, Api

api_blueprint = Blueprint("api", __name__, url_prefix="/api/v1")

api = Api(api_blueprint, version="1.0", title="Metacurate text processing API",
          description="API for accessing lexical knowledge, and text normalization.")

ns = api.namespace("lookup", description="Look-up semantically similar terms.")


# TODO how to style swagger?
@ns.route("/<string:term>", defaults={"num": 10})
@ns.route("/<string:term>/<int:num>")
@api.doc(responses={200: "The term to look-up is available in the lexicon",
                    400: "No term to look-up is specified",
                    404: "Term not in lexicon"},
         params={"term": "The term to look-up in the lexicon.",
                 "num": "The number of semantically similar terms to retrieve (optional)."})
class LookUp(Resource):

    def get(self, term, num):
        min = 1
        max = 50
        if num < min or num > max:
            api.abort(400,
                      "The number of semantically similar terms to retrieve must be "
                      "between {} and {} (inclusive)".format(
                          min, max))

        result = []
        similarities = []
        if term:
            term = term.lower().strip()
            if len(term) > 0:
                try:
                    similarities = g.metacurate_model.wv.most_similar(positive=term.replace(" ", "_"), topn=num)
                except KeyError:
                    api.abort(404, "The term {} is not available in the lexicon".format(term))
                for similarity in similarities:
                    text = str(similarity[0]).replace("_", " ")
                    score = str(round(similarity[1], 2))
                    result.append({"term": text, "score": score})
            else:
                api.abort(400, "No term to look-up is specified")
        else:
            api.abort(400, "No term to look-up is specified")
        return result

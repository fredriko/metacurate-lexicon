from flask import Blueprint, g
from flask_restplus import Resource, Api, fields

api_blueprint = Blueprint("api", __name__, url_prefix="/api/v1")

api = Api(api_blueprint, version="1.0", title="Metacurate text processing API",
          description="API for accessing lexical knowledge, and text normalization.")

ns = api.namespace("lookup", description="Look-up semantically similar terms.")

resource_fields = api.model('lookup response', {
    "term": fields.String(description="The term semantically similar to the one looked-up in the lexicon."),
    "similarity": fields.Float(description="The cosine similarity score of the input term and one to which this "
                                           "score is associated. Range: -1 to 1. The higher the value, the more "
                                           "semantically similar the two terms are.")
})


@ns.route("/<string:term>", defaults={"num": 10})
@ns.route("/<string:term>/<int:num>")
@api.doc(responses={200: "The term to look-up is available in the lexicon",
                    400: "No term to look-up is specified",
                    404: "Term not in lexicon"},
         params={"term": "The term to look-up in the lexicon.",
                 "num": "The number of semantically similar terms to retrieve (optional)."})
class LookUp(Resource):

    @ns.marshal_with(resource_fields, as_list=True)
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
                    similarities = g.metacurate_vectors.most_similar(positive=term.replace(" ", "_"), topn=num)
                except KeyError:
                    api.abort(404, "The term {} is not available in the lexicon".format(term))
                for similarity in similarities:
                    text = str(similarity[0]).replace("_", " ")
                    score = str(round(similarity[1], 2))
                    result.append({"term": text, "similarity": score})
            else:
                api.abort(400, "No term to look-up is specified")
        else:
            api.abort(400, "No term to look-up is specified")
        return result

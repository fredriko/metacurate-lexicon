from flask import Blueprint, g
from flask_restplus import Resource, Api, fields
from src.scripts.data_preparation import split_and_tokenize

api_blueprint = Blueprint("api", __name__, url_prefix="/api/v1")

api = Api(api_blueprint, version="1.0", title="Metacurate text processing API",
          description="API for accessing lexical knowledge, and text normalization.")

lookup_response_fields = api.model("lookup response", {
    "term": fields.String(description="The term semantically similar to the one looked-up in the lexicon."),
    "similarity": fields.Float(description="The cosine similarity score of the input term and one to which this "
                                           "score is associated. Range: -1 to 1. The higher the value, the more "
                                           "semantically similar the two terms are.")
})

tokenize_response_fields = api.model("tokenize response", {
    "sentences":
        fields.List(
            fields.List(
                fields.String(description="A token. Tokens containing undescore characters, "
                                          "i.e., _, are multi-word expressions: the underscores "
                                          "should be considered for removal before subsequent "
                                          "processing of the token.", required=True),
                description="A sentence."), description="A list of sentences.")
})

parser = api.parser()
parser.add_argument("text", type=str, required=True, help="The text to be tokenized.")


@api.route("/lookup/<string:term>", defaults={"num": 10})
@api.route("/lookup/<string:term>/<int:num>")
@api.doc(responses={200: "The term to look-up is available in the lexicon.",
                    400: "No term to look-up is specified in the request.",
                    404: "Term not in lexicon."},
         params={"term": "The term to look-up in the lexicon.",
                 "num": "The number of semantically similar terms to retrieve (optional)."})
class LookUp(Resource):

    @api.marshal_with(lookup_response_fields, as_list=True)
    def get(self, term, num):
        min = 1
        max = 50
        if num < min or num > max:
            api.abort(400,
                      "The number of semantically similar terms to retrieve must be "
                      "between {} and {} (inclusive).".format(
                          min, max))

        result = []
        similarities = []
        if term:
            term = term.lower().strip()
            if len(term) > 0:
                try:
                    similarities = g.metacurate_vectors.most_similar(positive=term.replace(" ", "_"), topn=num)
                except KeyError:
                    api.abort(404, "The term {} is not available in the lexicon.".format(term))
                for similarity in similarities:
                    text = str(similarity[0]).replace("_", " ")
                    score = str(round(similarity[1], 2))
                    result.append({"term": text, "similarity": score})
            else:
                api.abort(400, "No term to look-up is specified.")
        else:
            api.abort(400, "No term to look-up is specified.")
        return result


@api.route("/tokenize/")
@api.doc(responses={200: "Processing successful.",
                    400: "No text to tokenize is specified in the request."})
class Tokenize(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(tokenize_response_fields)
    def post(self):
        result = []
        arguments = parser.parse_args()
        text = arguments["text"]
        text = text.strip()
        if len(text) > 0:
            sentences = split_and_tokenize(text)
            for sentence in sentences:
                result.append(list(g.metacurate_trigrams[g.metacurate_bigrams[sentence]]))
            return {"sentences": result}, 200
        else:
            api.abort(400, "No text to tokenize is specified in the request.")

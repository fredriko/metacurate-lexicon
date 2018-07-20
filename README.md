# Metacurate Lexicon

## tl;dr
The metacurate lexicon, and the accompanying API, are the results of an investigation into the feasibility 
to deploy a web service that uses a reasonably large set of word embeddings to platform-as-a-service Heroku.

## verbose

The metacurate lexicon is available at 
[https://metacurate-lexicon.herokuapp.com/](https://metacurate-lexicon.herokuapp.com/) 
(it is running on a free dyno, so it takes 30 seconds or so to spin it up). 
It is a python/Flask web application that exposes interfaces (a web GUI and a RESTful API) for looking up 
semantically similar (multi-word) terms in a lexicon, as well as the appropriate pre-processing of raw text 
into sentences and term tokens. The word embeddings in the lexicon are generated by the gensim word2vec 
implementation, and the recognition of multi-word terms is based on gensim Phraser:s.

Here's a screenshot of looking up the term *word embedding* in the lexicon:

![first page of metacurate lexicon](src//static/metacurate-lexicon-word-embedding.png)


Here's a screenshot of the automatically generated API docmentation:

![the api documentation](src/static/metacurate-lexicon-api.png)

Upcoming features at [metacurate.io](https://metacurate.io) require access to a lexicon of semantically similar 
multi-word terms. Since metacurate.io is hosted on [heroku](https://www.heroku.com/), 
I wanted to find out whether the required semantic lexicon functionality can be deployed to heroku too,
without violating their application size constraints.

The answer is *yes*.

## How to run the web service locally

You will need:

* Python 3.6
* virutalenv 


* Create a virtual environment, let's call it *metacurate-lexicon*
* *pip install -r requirements.txt*

## How to deploy the web service to Heroku

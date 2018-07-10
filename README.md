# Todo
* Get and display information about the lexicon, e.g., how many entries, common multi-word expressions, etc.
* Add morphology to lookup results in API

# Data
The starting point of this endeavour is the data collected by [metacurate.io](https://metacurate.io): at the time of
the this project, there were around 7000 URLs collected specifically related to artificial intelligence, data science, 
natural language processing, deep learning, and all sorts of data-related issues. After pre-processing, the texts
amounted to some 280k sentences (longer than four tokens), and 6,35M words.

Now, that isn't all that much. We clearly need more data to be able to train reliable collocation models, as well
as semantic lexicons.


If you are logged into [webhose.io](https://webhose.io), you have access to a number of 
[free data sets](https://webhose.io/datasets/). The collocations, 
as well as some of the lexicon, are based on the following datasets:

* **English news articles.**
English news articles originated in the US from the top 1,000 (based on the ranking provided by Alexa) news sites,	*499,610 articles*,	Nov 2016
* **Technology news articles.**
English news articles dealing with technology,	*41,476 articles*,	Sep-Oct 2015
* **Random posts from online message boards.**
English posts originating from the US based forums & message boards with over 20 participants in the thread,	*1,806,440 posts*,	Mar 2017
* **Popular News articles.**
English news articles with at least 100 Facebook likes within 3 days of original post,	*170,882 articles*,	Feb-Mar 2017
* **Popular Blog posts.** 
English blog posts with at least 100 Facebook likes within 3 days of original post,	*87,510 posts*,	Feb-Mar 2017


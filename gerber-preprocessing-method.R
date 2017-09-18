#Preparing data using Professor Gerber's method

#Install (if necessary) and load the core tidyverse packages: ggplot2, tibble, tidyr, readr, purrr, and dplyr
library(tidyverse) 
library(XML)
library(tm)

#Read in files:
train <- read_csv("train.csv") #Read in the comma separated value data file for training the model
test <- read_csv("test.csv") #Read in the csv data file for testing the model
sample <- read_csv("sample.csv") #Read in the csv data file for sample submission (for reference)

#Get the content:
tweets = VCorpus(DataframeSource(train[,2]))

#Regular indexing returns a sub-corpus
inspect(tweets[1:2])

#Double indexing accesses actual documents
tweets[[1]]
tweets[[1]]$content

#Compute TF-IDF matrix and inspect sparsity
tweets.tfidf = DocumentTermMatrix(tweets, control = list(weighting = weightTfIdf))
tweets.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
#Sparsity is number of non-zero cells divided by number of zero cells

#Inspect sub-matrix:  first 5 documents and first 5 terms
tweets.tfidf[1:5,1:5]
as.matrix(tweets.tfidf[1:5,1:5])

##### Reducing Term Sparsity #####

#Clean up the corpus.
tweets.clean = tm_map(tweets, stripWhitespace)                          # remove extra whitespace
tweets.clean = tm_map(tweets.clean, removeNumbers)                      # remove numbers
tweets.clean = tm_map(tweets.clean, removePunctuation)                  # remove punctuation
tweets.clean = tm_map(tweets.clean, content_transformer(tolower))       # ignore case
tweets.clean = tm_map(tweets.clean, removeWords, stopwords("english"))  # remove stop words
tweets.clean = tm_map(tweets.clean, stemDocument)                       # stem all words

#Compare original content of document 1 with cleaned content
tweets[[1]]$content
tweets.clean[[1]]$content  # do we care about misspellings resulting from stemming?

#Recompute TF-IDF matrix
tweets.clean.tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf))

#Reinspect the first 5 documents and first 5 terms
tweets.clean.tfidf[1:5,1:5]
as.matrix(tweets.clean.tfidf[1:5,1:5])

#We've still got a very sparse document-term matrix. Remove sparse terms at various thresholds.
tfidf.99 = removeSparseTerms(tweets.clean.tfidf, 0.99)  #Remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.99
as.matrix(tfidf.99[1:5,1:5])

tfidf.70 = removeSparseTerms(tweets.clean.tfidf, 0.70)  #Remove terms that are absent from at least 70% of documents
tfidf.70
as.matrix(tfidf.70[1:2, 1:2])
tweets.clean[[1]]$content

#Which documents are most similar?
dtm.tfidf.99 = as.matrix(tfidf.99)
dtm.dist.matrix = as.matrix(dist(dtm.tfidf.99))
most.similar.documents = order(dtm.dist.matrix[1,], decreasing = FALSE)
tweets[[most.similar.documents[1]]]$content
tweets[[most.similar.documents[2]]]$content
tweets[[most.similar.documents[3]]]$content
tweets[[most.similar.documents[4]]]$content
tweets[[most.similar.documents[5]]]$content



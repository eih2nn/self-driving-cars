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

#Compute TF-IDF matrix and inspect sparsity
tweets.tfidf = DocumentTermMatrix(tweets, control = list(weighting = weightTfIdf))
tweets.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.

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

#Inspect the first 5 documents and first 5 terms
tweets.clean.tfidf[1:5,1:5]
as.matrix(tweets.clean.tfidf[1:5,1:5])

#We've still got a very sparse document-term matrix. Remove sparse terms at various thresholds.
tfidf.99 = removeSparseTerms(tweets.clean.tfidf, 0.99)  #Remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.99
as.matrix(tfidf.99[1:5,1:5])
dtm.tfidf.99 = as.matrix(tfidf.99)

df.99.scored <- data.frame(dtm.tfidf.99)
df.99.scored["SCORE"] <- train[,1]

colnames.99 <- as.list(colnames(df.99.scored))
colnames.99

lm.df.99 <- lm(SCORE~., 
data=df.99.scored)

summary(lm.df.99)

lm.df.99.2 <- lm(SCORE~cant+dont+excit+googl+insur+less+need+pedal+safer+save+saw+
                   soon+thing+wait+want+warn+wrong, #Select anything with significance
               data=df.99.scored)
summary(lm.df.99.2)

lm.df.99.3 <- lm(SCORE~cant+dont+excit+googl+insur+less+need+safer+
                   soon+thing+wait+want+warn+wrong, #Select anything with significance
                 data=df.99.scored)
summary(lm.df.99.3)

###### PREPARE TESTING SET ###### 

#Get the content:
test = VCorpus(DataframeSource(test[,2]))

#Compute TF-IDF matrix and inspect sparsity
test.tfidf = DocumentTermMatrix(test, control = list(weighting = weightTfIdf))
test.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.

##### Reducing Term Sparsity #####

#Clean up the corpus.
test.clean = tm_map(test, stripWhitespace)                          # remove extra whitespace
test.clean = tm_map(test.clean, removeNumbers)                      # remove numbers
test.clean = tm_map(test.clean, removePunctuation)                  # remove punctuation
test.clean = tm_map(test.clean, content_transformer(tolower))       # ignore case
test.clean = tm_map(test.clean, removeWords, stopwords("english"))  # remove stop words
test.clean = tm_map(test.clean, stemDocument)                       # stem all words

#Recompute TF-IDF matrix and convert to dataframe
test.clean.tfidf = DocumentTermMatrix(test.clean, control = list(weighting = weightTfIdf))
test.clean.tfidf = as.matrix(test.clean.tfidf)
df.test.preds <- data.frame(test.clean.tfidf)

mypreds <- data.frame(predict(lm.df.99.3, newdata = df.test.preds))
sentiment <- round(mypreds[,1],digits=0)
lm.preds <- as.data.frame(sentiment)
lm.preds[51,] = 5 #Change single 6 value to a 5

test <- read_csv("test.csv") #Read in the csv data file for testing the model
lm.preds["id"] = test[,1]
lm.preds <- lm.preds[c(2,1)] #Switch columns

write.table(lm.preds, file = "lm_car_tweets_eih.csv", row.names=F, sep=",") #Write out to a csv

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

# Use step function to find optimal set of predictors
# step(lm.df.99, direction = "both")

lm.optimal <- lm(formula = SCORE ~ accid + cant + car + come + cool + dont + 
                   excit + fbi + googl + hit + insur + just + less + need + 
                   pedal + safer + save + saw + soon + taxi + thing + think + 
                   truck + wait + want + warn + wheel + will + wrong + yes, 
                 data = df.99.scored)

summary(lm.optimal)

###### PREPARE TESTING SET ###### 

test2 <- rbind(test, c(1000, paste(colnames(df.99.scored), collapse = " ")))

#Get the content:
test2 = VCorpus(DataframeSource(test2[,2]))

#Compute TF-IDF matrix and inspect sparsity
test.tfidf = DocumentTermMatrix(test2, control = list(weighting = weightTfIdf))
test.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.

##### Reducing Term Sparsity #####

#Clean up the corpus.
test.clean = tm_map(test2, stripWhitespace)                          # remove extra whitespace
test.clean = tm_map(test.clean, removeNumbers)                      # remove numbers
test.clean = tm_map(test.clean, removePunctuation)                  # remove punctuation
test.clean = tm_map(test.clean, content_transformer(tolower))       # ignore case
test.clean = tm_map(test.clean, removeWords, stopwords("english"))  # remove stop words
test.clean = tm_map(test.clean, stemDocument)                       # stem all words

#Recompute TF-IDF matrix and convert to dataframe

# Make sure
test.clean.tfidf = DocumentTermMatrix(test.clean, control = list(weighting = weightTfIdf))
test.clean.tfidf = as.matrix(test.clean.tfidf)
df.test.preds <- data.frame(test.clean.tfidf)
df.test.preds <- df.test.preds[-nrow(df.test.preds),]

mypreds <- data.frame(predict(lm.optimal, newdata = df.test.preds))
sentiment <- round(mypreds[,1],digits=0)
lm.preds <- as.data.frame(sentiment)

# Change <1 to 1 and >5 to 5 
lm.preds[lm.preds > 5,] = 5
lm.preds[lm.preds < 1,] = 1

test <- read_csv("test.csv") #Read in the csv data file for testing the model
lm.preds["id"] = test[,1]
lm.preds <- lm.preds[c(2,1)] #Switch columns

write.table(lm.preds, file = "lm_car_tweets_bhg.csv", row.names=F, sep=",") #Write out to a csv

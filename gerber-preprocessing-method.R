#TEAM COMPETITION 3-1
#Elizabeth Homan
#Ben Greenawald
#Leelakrishna (Sai) Bollempalli

#Install (if necessary) and load the core tidyverse packages: ggplot2, tibble, tidyr, readr, purrr, and dplyr
library(tidyverse) 
library(XML)
library(tm)

#Read in files:
train <- read_csv("train.csv") #Read in the comma separated value data file for training the model
test <- read_csv("test.csv") #Read in the csv data file for testing the model
sample <- read_csv("sample.csv") #Read in the csv data file for sample submission (for reference)

### PREPARE/CLEAN TRAINING SET ###

#Get the content:
tweets = VCorpus(DataframeSource(train[,2]))

#Compute TF-IDF matrix and inspect sparsity
tweets.tfidf = DocumentTermMatrix(tweets, control = list(weighting = weightTfIdf))
tweets.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.

#Reducing Term Sparsity...
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

#Remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.99 = removeSparseTerms(tweets.clean.tfidf, 0.99)  
tfidf.99
as.matrix(tfidf.99[1:5,1:5])
dtm.tfidf.99 = as.matrix(tfidf.99)

df.99.scored <- data.frame(dtm.tfidf.99)
df.99.scored["SCORE"] <- train[,1]

###### PREPARE TESTING SET ###### 

#Get the content:
test = VCorpus(DataframeSource(test[,2]))

#Compute TF-IDF matrix and inspect sparsity
test.tfidf = DocumentTermMatrix(test, control = list(weighting = weightTfIdf))
test.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.

#Reducing Term Sparsity...
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

### CREATE BASIC LINEAR MODEL WITH CLEANED 99% TRAINING SET ###

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

## LM PREDICTIONS... ###

mypreds <- data.frame(predict(lm.df.99.3, newdata = df.test.preds))
sentiment <- round(mypreds[,1],digits=0)
lm.preds <- as.data.frame(sentiment)
lm.preds[51,] = 5 #Change single 6 value to a 5

test <- read_csv("test.csv") #Read in the csv data file for testing the model
lm.preds["id"] = test[,1]
lm.preds <- lm.preds[c(2,1)] #Switch columns

write.table(lm.preds, file = "lm_car_tweets_eih.csv", row.names=F, sep=",") #Write out to a csv
#KAGGLE SCORE = 0.68302

### KNN APPROACH (NOT FROM SCRATCH) ###

library(class)  
library(caret)

#Train a model using knn, with 20 runs of different K values
knn.fit <- train(SCORE~., data = df.99.scored, method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 15)
knn.fit

#k-Nearest Neighbors 
#981 samples
#126 predictors
#Pre-processing: centered (126), scaled (126) 
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 981, 981, 981, 981, 981, 981, ... 
#Resampling results across tuning parameters:
  
#  k   RMSE       Rsquared  
#5  0.8663321  0.01356836
#7  0.8354020  0.01216891
#9  0.8126616  0.01471686
#11  0.7980212  0.01811100
#13  0.7842202  0.02449978
#15  0.7771873  0.02669756
#17  0.7729275  0.02767401
#19  0.7687481  0.03047114
#21  0.7678358  0.02969977
#23  0.7668293  0.02969825
#25  0.7917816  0.02870664
#27  0.7903342  0.03115282
#29  0.7906914  0.03014757
#31  0.7911242  0.02967747
#33  0.7915552  0.02883294

#RMSE was used to select the optimal model using  the smallest value.
#The final value used for the model was k = 27.

#Use KNN model for test data and put predictions into a new dataframe
knn.preds <- predict(knn.fit, newdata = df.test.preds)
#ERROR - multiple objects not found

#Drop necessary columns from training dataframe and rerun knn.fit()
drops <- c("fbi","januari","univers")
df.99.drops <- df.99.scored[ , !(names(df.99.scored) %in% drops)]

knn.fit <- train(SCORE~., data = df.99.drops, method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 15)
knn.fit
#...RMSE was used to select the optimal model using  the smallest value.
#The final value used for the model was k = 27.

mypreds <- data.frame(predict(knn.fit, newdata = df.test.preds))
sentiment <- round(mypreds[,1],digits=0)
knn.preds <- as.data.frame(sentiment)
knn.preds["id"] <- test[,1] #Add ID values
knn.preds <- knn.preds[c(2,1)] #Switch column order

write.table(knn.preds, file = "knn_car_tweets_eih.csv", row.names=F, sep=",") #Write out to a csv
#OF NOTE: THIS DATAFRAME IS ALMOST ENTIRELY SENTIMENT = 3; IT IS NOT GOING TO BE A GOOD
#OPTION FOR SUBMISSION


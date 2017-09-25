#TEAM COMPETITION 3-1
#Elizabeth Homan
#Ben Greenawald
#Leelakrishna (Sai) Bollempalli

library(tm)
library(readr)
source("knn-from-scratch.R")
source("preprocess.R")
library(class) # ONLY USED FOR EXPLORING DIFFERENT KNN MODELS (because it is so much faster), 
# PREDICTIONS USE OUR MODEL AS SPECIFIED BY THE ASSIGNMENT
library(dplyr)

##### Read in the data
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# Check on distribution on numbers
train2 <- train %>% 
  group_by(sentiment) %>% 
  summarize(n = n())

# Run the KNN

set.seed(3)
# Use LOOCV to find range of good k-values
preds <- clean_data(train, 0.99, F, F, F)
k <- seq(1, as.integer(sqrt(nrow(preds))) + 4, by = 2)
reses <- vector(mode = "numeric", length = length(k))
for(i in 1:length(k)){
  reses[i] <- knn.loocv(preds, train$sentiment, k = k[i])
  print(i)
}
reses 
# Gives a range of k of 11-33

# Using range of k's, run kfold to narrow even further
preds <- clean_data(train, 0.99, F, F, F)
k <- seq(11, 33, by = 2)
reses <- vector(mode = "numeric", length = length(k))
for(i in 1:length(k)){
  reses[i] <- knn.kfolds(preds, as.factor(train$sentiment), k = k[i])
  print(i)
}
reses 
# Gives best k-value of ~25 so we will just use 25

# Optimize the sparcity using optimal k
sparc <- c(0.95, 0.96, 0.975, 0.985, 0.99)
reses <- vector(mode = "numeric", length = length(sparc))
for(i in 1:length(sparc)){
  preds <- clean_data(train, sparc[i], F, T, F)
  p <- knn.cv(preds, as.factor(train$sentiment), k = 25)
  reses[i] <- sum(p == train$sentiment)/length(train$sentiment)
}
reses
# Gives optimal sparcity of 0.975-0.99 so we will continue with 
# sparcity 0.975 to reduce dimensionality unless otherwise specified

# See if keeping stopwords helps (first is stopwords removed)
preds <- clean_data(train, 0.975, F, F, F)
p <- knn.cv(preds, as.factor(train$sentiment), k = 25)
sum(p == train$sentiment)/length(train$sentiment)
# 0.6136595

preds <- clean_data(train, 0.975, F, T, F)
p <- knn.cv(preds, as.factor(train$sentiment), k = 25)
sum(p == train$sentiment)/length(train$sentiment)
# 0.6146789
# Inconclusive results

# See if adding additional features helps (first is without adding)
preds <- clean_data(train, 0.975, F, F, F)
p <- knn.cv(preds, as.factor(train$sentiment), k = 25)
sum(p == train$sentiment)/length(train$sentiment)
# 0.6146789

preds <- clean_data(train, 0.975, F, F, T)
p <- knn.cv(preds, as.factor(train$sentiment), k = 25)
sum(p == train$sentiment)/length(train$sentiment)
# 0.6136595
# Inconclusive results

# Play around with different distributions
train2 <- expand_data(train, c(3, 1, 1, 1, 2))
preds <- clean_data(train2, 0.995, F, F, F)
p <- class::knn.cv(preds, as.factor(train2$sentiment), k = 19)
sum(p == train2$sentiment)/length(train2$sentiment)
# Many different values for the distribution were tried, 
# this only seemed to make things worse.

# Play around with ngram analysis
# Change sparcity to include more of the bigrams
preds <- clean_data(train, 0.99, F, F, T, ngram = T)
p <- knn.kfolds(preds, as.factor(train$sentiment), k = 19, cosine = T)
sum(p == train$sentiment)/length(train$sentiment)
# 0.6156983
# Bigram seems to help a little bit

# Train on new data

# Start by preprocessing data based on rules discovered above
train.data <- clean_data(train, 0.99, filter_symbol = F, stop_words = F, 
                         extra = F, ngram = T)
corpus <- colnames(train.data)[!(colnames(train.data) %in% c("num_at", "num_exlaim",
                                                                    "num_hash", "num_question", "odd_char"))]

# Process the test data in the same way
test.data <- clean_data(test, 0.99, filter_symbol = F, stop_words = F, 
                        extra = F, ngram = T, dict = corpus)

# Add the extra features
train.data <- cbind(train.data, feature_extract(train))
test.data <- cbind(test.data, feature_extract(test))

predicts <- knn.R(train = train.data, test = test.data, cl = train$sentiment, k = 25, cosine = F)

results <- cbind("id" = test$id, "sentiment" = predicts)

write_csv(data.frame(results), "Predictions/predictions_knn_final_ben.csv")

# The final accuracy (including the private test set) score of the K-NN model is 0.66530

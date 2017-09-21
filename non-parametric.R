library(tm)
library(readr)
source("knn from scratch.R")
source("preprocess.R")
library(MASS) # Used for lda
library(class)
library(dplyr)
library(tau)

##### Read in the data
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# Check on distribution on numbers
train2 <- train %>% 
  group_by(sentiment) %>% 
  summarize(n = n())

# Run the KNN

# Use LOOCV to find range of good k-values
preds <- clean_data(train, 0.99, F, F, F)
k <- seq(1, as.integer(sqrt(nrow(preds))) + 4, by = 2)
reses <- vector(mode = "numeric", length = length(k))
for(i in 1:length(k)){
  reses[i] <- knn.loocv(preds, train$sentiment, k = k[i])
  print(i)
  # reses[i] <- sum(p == train$sentiment)/length(train$sentiment)
}
reses 

# Gives a range of k of 11-33

# Using rang eof k's, run kfold to narrow even further
preds <- clean_data(train, 0.99, F, F, F)
k <- seq(11, 33, by = 2)
reses <- vector(mode = "numeric", length = length(k))
for(i in 1:length(k)){
  p <- knn.cv(preds, as.factor(train$sentiment), k = k[i])
  reses[i] <- sum(p == train$sentiment)/length(train$sentiment)
}
reses 
# Gives best k-value of 25

# Optimize the sparcity using optimal k
sparc <- c(0.95, 0.96, 0.975, 0.985, 0.99)
reses <- vector(mode = "numeric", length = length(sparc))
for(i in 1:length(sparc)){
  preds <- clean_data(train, sparc[i], F, T, F)
  p <- knn.cv(preds, as.factor(train$sentiment), k = 25)
  reses[i] <- sum(p == train$sentiment)/length(train$sentiment)
}
reses
# Gives optimal sparcity of 0.975

# See if keeping stopwords helps
preds <- clean_data(train, 0.975, F, F, F)
p <- knn.cv(preds, as.factor(train$sentiment), k = 19)
sum(p == train$sentiment)/length(train$sentiment)

preds <- clean_data(train, 0.975, F, T, F)
p <- knn.cv(preds, as.factor(train$sentiment), k = 19)
sum(p == train$sentiment)/length(train$sentiment)
# Removing stopwords helps

# See if adding additional features helps
preds <- clean_data(train, 0.975, F, T, F)
p <- knn.cv(preds, as.factor(train$sentiment), k = 19)
sum(p == train$sentiment)/length(train$sentiment)

preds <- clean_data(train, 0.975, F, T, T)
p <- knn.cv(preds, as.factor(train$sentiment), k = 19)
sum(p == train$sentiment)/length(train$sentiment)
# Adding features does help

# Play around with different distributions
train2 <- expand_data(train, c(3, 1, 1, 1, 2))
preds <- clean_data(train2, 0.995, F, T, F)
p <- class::knn.cv(preds, as.factor(train2$sentiment), k = 19)
sum(p == train2$sentiment)/length(train2$sentiment)

# Play around with ngram analysis
train2 <- expand_data(train, c(3, 1, 1, 1, 2))
preds <- clean_data(train, 0.99, F, T, F, ngram = T)
p <- class::knn.cv(preds, as.factor(train$sentiment), k = 19)
sum(p == train$sentiment)/length(train$sentiment)

# Train on new data

# Start by preprocessing data based on rules discovered above
train.data <- clean_data(train, 0.975, F, T, F)

# Process the test data in the same way
test.data <- rbind(test, c(1000, paste(colnames(train.data), collapse = " ")))
test.data <- clean_data(test.data, 0.975, F, T, F, colnames(train.data))

# Add the extra features
train.data <- cbind(train.data, feature_extract(train))
test.data <- cbind(test.data, feature_extract(test))

predicts <- knn(train = train.data, test = test.data, cl = train$sentiment, k = 17)

results <- cbind("id" = test$id, "sentiment" = predicts)

write_csv(data.frame(results), "predictions_knn_ben.csv")

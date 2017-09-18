library(tm)
library(readr)
source("knn.R")
source("preprocess.R")
library(MASS) # Used for lda
library(class)

##### Read in the data
train <- read_csv("train.csv")

# Run the KNN

# Find ideal K with baseline values
preds <- clean_data(train, 0.99, F, T, F)
k <- seq(1, 2 * as.integer(sqrt(nrow(preds))), by = 2)
reses <- vector(mode = "numeric", length = length(k))
for(i in 1:length(k)){
  p <- knn.cv(preds, as.factor(train$sentiment), k = k[i])
  reses[i] <- sum(p == train$sentiment)/length(train$sentiment)
}
reses 

# Gives a best k of 19, 21, 23

# Optimize the sparcity using optimal k
sparc <- c(0.95, 0.96, 0.975, 0.985, 0.99)
reses <- vector(mode = "numeric", length = length(sparc))
for(i in 1:length(sparc)){
  preds <- clean_data(train, sparc[i], F, T, F)
  p <- knn.cv(preds, as.factor(train$sentiment), k = 19)
  reses[i] <- sum(p == train$sentiment)/length(train$sentiment)
}
reses
# Gives optimal sparcity of 0.975

# See if filtering symbols helps
preds <- clean_data(train, 0.975, F, T, F)
p <- knn.cv(preds, as.factor(train$sentiment), k = 19)
sum(p == train$sentiment)/length(train$sentiment)

preds <- clean_data(train, 0.975, T, T, F)
p <- knn.cv(preds, as.factor(train$sentiment), k = 19)
sum(p == train$sentiment)/length(train$sentiment)
# Keeping symbols seems to help

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
# Adding features does not help

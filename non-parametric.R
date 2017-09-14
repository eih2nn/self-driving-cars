# Apply the non-parametric KNN approach

source("knn.R")
source("preprocess.R")
library(tidytext)
library(dplyr)
library(ggplot2)
library(tm)

train <- read_csv("train.csv")

train_mod <- feature_extract(train)
ks <- seq(1, sqrt(nrow(train_mod)), by = 4)
reses <- vector(mode = "numeric", length = length(ks))
for(i in 1:length(ks)){
  reses[i] <- knn.loocv(train_mod[, -c(1, 2)], train_mod[, 1], k = ks[i])
}

fold <- 5
ks <- seq(1, sqrt(nrow(train_mod) - nrow(train_mod)/fold), by = 4)
reses.kfold <- vector(mode = "numeric", length = length(ks))

for(i in 1:length(ks)){
 reses.kfold[i] <- knn.kfolds(train_mod[, -c(1, 2)], train_mod[, 1], k = ks[i]) 
}

sample <- sample(1:nrow(train_mod), size = 0.8 * nrow(train_mod))

train.data <- train_mod[sample, -c(1,2)]
train.cl <- train_mod[sample, 1]
test.data <- train_mod[-sample, -c(1, 2)]
true.cl <- train_mod[-sample, 1]

preds <- knn.R(train.data, test.data, train.cl, k = 3)
sum(preds == true.cl)/nrow(true.cl)

# Play with the tidytext package
tweet_words <-  train %>% 
  unnest_tokens(word, text) %>%
  count(sentiment, word, sort = TRUE) %>%
  ungroup()

total_words <- tweet_words %>% 
  group_by(sentiment) %>% 
  summarize(total = sum(n))

tweet_words <- left_join(tweet_words, total_words)

tweet_words <- tweet_words %>%
  bind_tf_idf(word, sentiment, n)

# Play with the tm package
c <- Corpus(VectorSource(train_mod$text))

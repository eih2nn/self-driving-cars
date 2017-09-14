# Script to preprocess the text data

library(dplyr)
library(readr)
library(stringr)

# Read in the train data

train <- read_csv("train.csv")

feature_extract <- function(train){
  train_mod <- mutate(odd_char = str_count(text, "Â"), train)
  train_mod <- mutate(str_len = str_length(text), train_mod)
  train_mod["num_words"] <- unlist(lapply(train_mod$text, num_words))
  train_mod["num_hash"] <- unlist(lapply(train_mod$text, num_hashtag))
  train_mod["num_at"] <- unlist(lapply(train_mod$text, num_at))
  return(train_mod)
}


num_words <- function(x){
  return(length(unlist(str_split(x, " "))))
}

num_hashtag <- function(x){
  res <- unlist(str_split(x, " "))
  return(sum(grepl("@", res)))
}

num_at <- function(x){
  res <- unlist(str_split(x, " "))
  return(sum(grepl("#", res)))
}

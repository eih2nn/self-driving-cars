# Script to preprocess the text data

library(dplyr)
library(readr)
library(stringr)
library(tm)

clean_data <- function(train, sparcity, stop_words = T, extra = T){
  train2 <- train
  train2$text <- remove_at(train2$text)
  train2$text <- remove_symbols(train2)
  ##### Constructing TF-IDF Matrices #####
  tweets <- Corpus(VectorSource(train2$text))
  
  # regular indexing returns a sub-corpus
  inspect(tweets[1:2])
  
  # double indexing accesses actual documents
  tweets[[1]]
  tweets[[1]]$content
  
  ##### Reducing Term Sparsity #####
  
  # there's a lot in the documents that we don't care about. clean up the corpus.
  tweets.clean = tm_map(tweets, stripWhitespace)                          # remove extra whitespace
  tweets.clean = tm_map(tweets.clean, removeNumbers)                      # remove numbers
  tweets.clean = tm_map(tweets.clean, removePunctuation)                  # remove punctuation
  tweets.clean = tm_map(tweets.clean, content_transformer(tolower))       # ignore case
  # tweets.clean = tm_map(tweets.clean, removeWords, stopwords("english"))  # remove stop words
  tweets.clean = tm_map(tweets.clean, stemDocument)                       # stem all words
  
  # compare original content of document 1 with cleaned content
  tweets[[1]]$content
  tweets.clean[[1]]$content  # do we care about misspellings resulting from stemming?
  
  ret.tweets <- as.data.frame(tweets.clean$content)
  
  # recompute TF-IDF matrix
  tweets.clean.tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf))
  
  # reinspect the first 5 documents and first 5 terms
  tweets.clean.tfidf[1:5,1:5]
  as.matrix(tweets.clean.tfidf[1:5,1:5])
  
  # we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
  tfidf = removeSparseTerms(tweets.clean.tfidf, 0.975) 
  tfidf
  
  dtm.tfidf = as.matrix(tfidf)
  
  if(extra){
    extra_features <- feature_extract(train)
    extra_features$sentiment <- NULL 
    extra_features$text <- NULL 
    preds <- cbind(dtm.tfidf, extra_features)
    preds <- as.data.frame(preds)
    return(preds)
  }else{
    return(dtm.tfidf)
  }
  
}

# Read in the train data

feature_extract <- function(train){
  train_mod <- mutate(odd_char = str_count(text, "[Ì¢???âÂåüèÏ???Û¡ÂsteÃ]"), train)
  # train_mod <- mutate(str_len = str_length(text), train_mod)
  # train_mod["num_words"] <- unlist(lapply(train_mod$text, num_words))
  train_mod["num_hash"] <- unlist(lapply(train_mod$text, num_hashtag))
  train_mod["num_at"] <- unlist(lapply(train_mod$text, num_at))
  train_mod <- mutate("num_exlaim" = str_count(text, "!"), train_mod)
  train_mod <- mutate("num_question" = str_count(text, "\\?"), train_mod)
  
  # Scale all features
  train_mod$odd_char <- train_mod$odd_char/max(train_mod$odd_char)
  train_mod["num_hash"] <- train_mod["num_hash"]/max(train_mod["num_hash"])
  train_mod["num_at"] <- train_mod["num_at"]/max(train_mod["num_at"])
  train_mod$num_exlaim <- train_mod$num_exlaim/max(train_mod$num_exlaim)
  train_mod$num_question <- train_mod$num_question/max(train_mod$num_question)
  
  return(train_mod)
}

remove_at <- function(x){
    x <- lapply(x, function(y){gsub("@\\w+ *", "", y)})
}

remove_symbols <- function(x){
  return(str_replace_all(x$text, "[Ì¢???âÂåüèÏ???Û¡ÂÃ???Ò]", ""))
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

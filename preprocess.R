# Script to preprocess the text data

library(dplyr)
library(readr)
library(stringr)
library(tm)
library(zoo) # Used for the coredata function
library(tau) # Used to tokenize into n-grams
library(RWeka)

# Clean the data based on the Professor Gerbers preprocessing script
# @param train: unprocessed data directly from csv file
# @param sparcity: sparcity threshold for the weighted matrix
# @param filter_symbol: boolean specifying whether to remove '@' symbols and
#   other oddly encoded symbols
# @param stop_words: boolean specifying whether to remove stop words or not
# @param extra: boolean specifying whether to extract additional features from 
#   the data
# @param dict: list representing all words in the train weighted matrix
# @param ngram: boolean telling whether to use bigram analysis
# @param n: Number of grams to use in ngram
clean_data <- function(train, sparcity = 0.99, filter_symbol = F, stop_words = F, extra = F, dict = NULL, ngram = F, n = 2){
  train2 <- train
  if(filter_symbol){
    train2$text <- remove_at(train2$text)
    train2$text <- remove_symbols(train2)
  }
  
  ##### Constructing TF-IDF Matrices #####
  tweets <- VCorpus(VectorSource(train2$text))

  ##### Reducing Term Sparsity #####
  
  # there's a lot in the documents that we don't care about. clean up the corpus.
  tweets.clean = tm_map(tweets, stripWhitespace)                          # remove extra whitespace
  tweets.clean = tm_map(tweets.clean, removeNumbers)                      # remove numbers
  tweets.clean = tm_map(tweets.clean, removePunctuation)                  # remove punctuation
  tweets.clean = tm_map(tweets.clean, content_transformer(tolower))       # ignore case
  if(!stop_words){
    tweets.clean = tm_map(tweets.clean, removeWords, stopwords("english"))  # remove stop words
  }
  tweets.clean = tm_map(tweets.clean, stemDocument)                       # stem all words
  
  if(ngram){
    # Tokenizer for ngrams
    BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = n))
    
    # recompute TF-IDF matrix
    if(is.null(dict)){
      tweets.clean.tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf,
                                                                           tokenize = BigramTokenizer))
      tfidf = removeSparseTerms(tweets.clean.tfidf, sparcity) 
    }else{
      tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf, 
                                                              dictionary = dict,
                                                              tokenize = BigramTokenizer))
    }
  }else{
    # recompute TF-IDF matrix
    if(is.null(dict)){
      tweets.clean.tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf))
      tfidf = removeSparseTerms(tweets.clean.tfidf, sparcity) 
    }else{
      tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf, 
                                                              dictionary = dict))
    }
  }
 
  
  # we've still got a very sparse document-term matrix. remove sparse terms at various thresholds
  
  dtm.tfidf = as.data.frame(as.matrix(tfidf))
  
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

# Extract additional, potentially useful, features from the data
feature_extract <- function(train){
 # train_mod <- mutate(odd_char = str_count(text, "[Ì¢???âÂåüèÏ???Û¡ÂsteÃ]"), train)
  train_mod <- train
  train_mod["num_hash"] <- unlist(lapply(train_mod$text, num_hashtag))
  train_mod["num_at"] <- unlist(lapply(train_mod$text, num_at))
  train_mod <- mutate("num_exlaim" = str_count(text, "!"), train_mod)
  train_mod <- mutate("num_question" = str_count(text, "\\?"), train_mod)
  
  # Scale all features
 # train_mod$odd_char <- train_mod$odd_char/max(train_mod$odd_char)
  train_mod["num_hash"] <- train_mod["num_hash"]/max(train_mod["num_hash"])
  train_mod["num_at"] <- train_mod["num_at"]/max(train_mod["num_at"])
  train_mod$num_exlaim <- train_mod$num_exlaim/max(train_mod$num_exlaim)
  train_mod$num_question <- train_mod$num_question/max(train_mod$num_question)
  
  train_mod$text <- NULL
  train_mod$sentiment <- NULL
  train_mod$id <- NULL
  return(train_mod)
}

# Removes any word starting with the '@' symbol
remove_at <- function(x){
    x <- lapply(x, function(y){gsub("@\\w+ *", "", y)})
}

# Remove certain oddly encoded symbols
# remove_symbols <- function(x){
#   return(str_replace_all(x$text, "[Ì¢???âÂåüèÏ???Û¡ÂÃ???Ò]", ""))
# }

# Return the number of words in a string
num_words <- function(x){
  return(length(unlist(str_split(x, " "))))
}

# Returns the numbers of hasttag uses in a string
num_hashtag <- function(x){
  res <- unlist(str_split(x, " "))
  return(sum(grepl("@", res)))
}

# Return the number of '@' symbols used
num_at <- function(x){
  res <- unlist(str_split(x, " "))
  return(sum(grepl("#", res)))
}

# Expand the data to normalize the classes
expand_data <- function(x, distribution){
  for(i in 1:5){
    temp <- x[x$sentiment == i, ]
    expand <- do.call(rbind, replicate(distribution[i], coredata(temp), simplify = FALSE))
    if(i == 1){
      result <- expand
    }else{
      result <- rbind(result, expand)
    }
  }
  return(result)
}

# Token data into n-gram tokens, comes from https://stackoverflow.com/questions/8898521/finding-2-3-word-phrases-using-r-tm-package
tokenize_ngrams <- function(x, n=1) return(rownames(as.data.frame(unclass(textcnt(x,method="string",n=n)))))


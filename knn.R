# Implementation of the KNN algorithm for classification

# Results results can differ slightly from other implementations
# due to how ties are broken when dealing with distance and class selection

# Wrapper function for the knn implementation
# @param train: Vectors of training predictors
# @param test: Vectors for unseen data
# @param class: Corresponding class of training predictors
knn.R <- function(train, test, class, k = 1){
  train.full <- cbind(train, class)
  preds <- unname(unlist(apply(test, 1, knn.classify, train = train.full, k = k)))
  return(preds)
}

# Classifies a single test case based on training data
# @param train: training data with the response as the final column
# @param test.case: single vector of unseen response
knn.classify <- function(test.case, train, k){
  cl <- train[, ncol(train)]
  train <- train[, -ncol(train)]
  train["dist"] <- apply(train, 1, function(x){sqrt(sum((x - test.case)^2))})
  train <- cbind(train, cl)
  train <- head(train[order(train$dist), ], n = k)
  unique_classes <- unique(train$cl)
  return(unique_classes[which.max(tabulate(match(train$cl, unique_classes)))])
}



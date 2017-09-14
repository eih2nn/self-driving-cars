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

knn.loocv <- function(train, class, k = 1){
  accuracy <- vector(mode="numeric", length = nrow(train))
  train <- cbind(train, class)
  for(i in 1:nrow(train)){
    train.data <- train[-i, -ncol(train)]
    train.class <- train[-i, ncol(train)]
    test.data <- train[i, -ncol(train)]
    true.class <- train[i, ncol(train)]
    pred <- knn.R(train = train.data, test = test.data, class = train.class, k = k)
    accuracy[i] <- pred == true.class
  }
  
  return(mean(accuracy))
}

partition_size <- function(size, folds){
  mod <- size %% folds  
  ret <- rep(floor(size / folds), folds - mod)
  ret <- c(ret, rep(ceiling(size/folds), mod))
  return(ret)
}

knn.kfolds <- function(train, class, k = 1, folds = 5){
  partition_sizes <- partition_size(nrow(train), folds)
  kfolds <- list()
  indices <- 1:nrow(train)
  bounds <- 1:folds
  for(i in bounds){
    kfolds[[i]] <- sample(indices, size = partition_sizes[i])
    indices <- setdiff(indices, kfolds[i])
  }
  
  train <-cbind(train, class)
  accuracy <- vector(mode="numeric", length = folds)
  for(i in bounds){
    test_indices <- unlist(kfolds[i])
    
    train.data <- train[-test_indices, -ncol(train)]
    train.class <- train[-test_indices, ncol(train)]
    test.data <- train[test_indices, -ncol(train)]
    true.class <- train[test_indices, ncol(train)]
    pred <- knn.R(train = train.data, test = test.data, class = train.class, k = k)
    accuracy[i] <- mean(sum(pred == true.class)/length(pred))
  }
  
  return(mean(accuracy))
}


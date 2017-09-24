#TEAM COMPETITION 3-1
#Elizabeth Homan
#Ben Greenawald
#Leelakrishna (Sai) Bollempalli


# Implementation of the KNN algorithm for classification

# Results results can differ slightly from other implementations
# due to how ties are broken when dealing with distance and class selection

# Wrapper function for the knn implementation
# @param train: Vectors of training predictors
# @param test: Vectors for unseen data
# @param class: Corresponding class of training predictors
# @param cosine: Set as true to use the cosine distance between points
knn.R <- function(train, test, class, k = 1, cosine = F){
  train.full <- cbind(train, class)
  preds <- unname(unlist(apply(test, 1, knn.classify, train = train.full, k = k, cosine = cosine)))
  return(preds)
}

# Classifies a single test case based on training data
# @param train: training data with the response as the final column
# @param test.case: single vector of unseen response
knn.classify <- function(test.case, train, k, cosine = F){
  cl <- train[, ncol(train)]
  train <- train[, -ncol(train)]
  if(cosine){
    res <- apply(train, 1, distance_cosine, test.case =  test.case)
  }else{
    res <- apply(train, 1, distance, test.case =  test.case)
  }
  train <- cbind(train, "dist" = res)
  train <- as.data.frame(cbind(train, cl))
  train <- head(train[order(train$dist), ], n = k)
  unique_classes <- unique(train$cl)
  return(unique_classes[which.max(tabulate(match(train$cl, unique_classes)))])
}

distance <- function(x, test.case){
  ret <- sqrt(sum((x - test.case)^2))
  return(ret)
}

distance_cosine <- function(x, test.case){
  ret <- sum(x * test.case)/(sqrt(sum(x^2))*sqrt(sum(test.case^2)))
  return(ret)
}
knn.loocv <- function(train, class, k = 1, cosine = F){
  accuracy <- vector(mode="numeric", length = nrow(train))
  train <- cbind(train, class)
  for(i in 1:nrow(train)){
    train.data <- train[-i, -ncol(train)]
    train.class <- train[-i, ncol(train)]
    test.data <- train[i, -ncol(train)]
    true.class <- train[i, ncol(train)]
    pred <- knn.R(train = train.data, test = test.data, class = train.class, k = k, cosine = cosine)
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

knn.kfolds <- function(train, class, k = 1, folds = 5, cosine = F){
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
    pred <- knn.R(train = train.data, test = test.data, class = train.class, k = k, cosine = cosine)
    accuracy[i] <- mean(sum(pred == true.class)/length(pred))
  }
  
  return(mean(accuracy))
}


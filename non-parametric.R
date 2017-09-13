# Apply the non-parametric KNN approach

source("knn.R")
source("preprocess.R")

train <- read_csv("train.csv")

train_mod <- feature_extract(train)

sample <- sample(1:nrow(train_mod), size = 0.8 * nrow(train_mod))

train.data <- train_mod[sample, -c(1,2)]
train.cl <- train_mod[sample, 1]
test.data <- train_mod[-sample, -c(1, 2)]
true.cl <- train_mod[-sample, 1]

preds <- knn.R(train.data, test.data, train.cl, k = 3)
sum(preds == true.cl)/nrow(true.cl)

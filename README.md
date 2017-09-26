# Twitter Sentiment Analysis
Kaggle In Class Competition re: Twitter Sentiment Analysis of Self-Driving Cars

Team: Competition 3-1s

### Important Files

This repository contains many files used throughout this competition, but the important files are as follows:

* preprocess.R: File containing the functions used in the preprocessing of the data
* knn-from-scratch.R: File containing our implementation of the KNN classification algorithm
* parametric.R: File that contains code for our parametric models used in the competition
* non-parametric.R: File that contains our non-parametric (KNN) models used in this competition
* Predictions/: A directory containing all of our predictions from throughout the competition



### Group Roles

All group members participated in all parts of this project, but the primary responsibilities were distributed as follows:

* Elizabeth Homan: Github Coordinator and primary contributor to the linear analysis
* Sai Bollempalli: Exploratory data analysis and non-parametric analysis
* Ben Greenawald: Implemented the KNN algorithm and the preprocessing script



### Preprocessing

All of the preprocessing took place in a single R script that was then sourced in all other files. The principle function in the script was *clean_data* that took in raw data straight from the CSV and performed all necessary preprocessing, returning a data frame of features. This function takes in a variety of parameters which control how the data should be preprocessed. Some of these options include the option to keep stop words in, the ability to use n-gram tokenization, and the ability to specify the sparcity threshold for our feature matrix. A further option affords the ability to add some additional features as described below. 



### Feature Extraction

A principle task in this assignment was to discover the best features to extract from the tweets. At first, we did a simple unigram using TFIDF weights. We did experiment with using different weighting schemes using the *weigthSmart* function, but no improvements were seen from this inquiry, so we went back to using only TFIDF. In order to improve our analysis, we added some additional features that took into account the fact that we were analyzing tweets and the fact that we were trying to predict sentiment. The first two of these features were how many '@' the tweets used (or how many other twitter users the tweets was directed at) and the number of '#', indicating the number of hastags the user had in their tweet. The logic behind these features is that they tell us some meta information about the tweet being analyzed. The second two additional features were the numbers of exclamation points and the number of question marks. The logic here was that even though we wanted to remove punctuation, these particular punctuation marks can be indicative of sentiment. After seeing marginal improvement with these features, we extended the analysis to bigram analysis and saw a slight improvement with this as well.

### Non-Parametric Approach

#### KNN from scratch

As per the assignment, we impemented KNN from scratch for our non-parametric analysis. This code resides in the file *knn-from-scratch.R*. The principle function in this script in *knn.R* which takes in the exact same parameters in the *knn* function from the class package plus one additional parameter *cosine* which specifies to whether or not to use the cosine distance between vectors (default is euclidean). Two addtional wrapper functions were implemented that performed k-fold analysis on the train data, *knn.kfolds* and leave-one-out cross validation, *knn.loocv*.

*Note, that a good amount of the exploratory analysis done for this approach used the class package for knn analysis. This is because their implementation was much faster than ours, but for the LOOCV and the final predictions, our own implementation was used*. 

#### KNN Analysis

As specified, KNN was used for our non-parametric approach. The first step was to choose what *k* would work best for our analysis. We used our implementation of knn loocv to begin our search for a good *k*. This gave a pretty large range of *k* that performed comparably, so using that range, we re-ran the cross validation using k-fold (our implementation). This landed us on a reasonable k estimate of 25. Next, we wanted to find a good level of sparcity for our tfidf matrix. Using k-folds CV (the class implementation for speed), we found the optimal level of sparcity to be around the range 97.5%-99%. We then wanted to check both if removing stopwords helped and if adding additional features as described in the **Feature Extraction** section helped. Both results were unclear because the model seemed to be picking only 3's which makes some sense because they were the most dominant class. Rectifying this issue required using lower k, which caused the train set accuracy to drop considerably, so we decided to stick with our higher value of k. One thing we did to try and fix this issue was to create duplicate entries of the underrepresented sentiment categories. Experimenting with various different levels of duplication consistently showed a drop in train accuracy, so this was abandoned. Some experimenting was done with using the cosine distance as the distance between vectors for knn, but this was largely unsuccesful. In the end, the bigram analysis with additional features, stop words removed, and sparcity 0.99 produced our final, and best, model results, even though the predictions were mostly all 3's.

### Parametric Approach

The parametric approach presented it's own set of issues. Namely, that linear regression is not usually used for classification tasks. With this in mind, we started out by using parametric linear models normally more suited for the task. The two models we explored were LDA and multinomial logistic regression. Without going into great detail, both of these models performed poorly, both on internal cross validation and submitted tries. We then moved onto trying regular multiple linear regression and rounding our resulting predictors to fit our outcome space. Instead of cross validation, adjusted R^2 was the primary metric for model analysis. This presents an obvious problem, that our outcome space is 5 value categorical outcome, which will result in low adjusted R^2. We accepted this limitation and decided to use adjusted R^2 to compare models, since all models share the same problem, rather than use adjusted R^2 as an accurate representation of model fit, like it is normally used. It should also be noted that a higher sparcity threshold was used (0.99) since insignificant predictors could just be removed. For an initial run, a model was created using the tf-idf weighted terms as features. With a model this big, there were a great deal of insignificant predictors. Because the feature space was so highly dimensional, removing one variable at a time, as is the statistically correct procedure, was not viable. Features were removed in batch format for this run (remove all significant predictors, rerun the model, remove all significant predictors, ect.). This models predictions were submitted as a test, but the results were relatively good, outperforming our KNN model at the time. In order to be statistically valid, we went back and used the *step* function to perform bidirectional feature selection, but the process was slow. Further, the results were not nearly as good as when the features were removed in batch format. Because of this, the batch removal of feature selection as described above was used for the rest of the models. After testing regular tf-idf, the process was repeated with first adding the additional features described in the **Feature Selection**, and after that, using additional features with bigrams and well as unigrams. In the end, the model that produced the best predictors was the bigram model with additional features, selected using the batch feature removal technique.

### Final Results

Our best linear regression model produced an accuracy of 0.64488
Our best K-NN model produced an accuracy of 0.6653

### Discussion

Interestingly, bigrams added little value in both linear regression and K-NN model. Selecting predictors for linear regression by plotting sentiment score against individual variables was not an effective approach for this text mining problem. This is probably because of the sparsity of the data frame especially considering we had 1000 tweets to train the model with. Further, the uneven distribution of the classes gave a great deal of trouble to the KNN which really only guessed 3's (the most dominate class by far).

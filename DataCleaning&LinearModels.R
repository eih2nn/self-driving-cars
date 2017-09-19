#TEAM COMPETITION 3-1
#Elizabeth Homan
#Ben Greenawald
#Leelakrishna (Sai) Bollempalli

#Load the core tidyverse packages (ggplot2, tibble, tidyr, readr, purrr, and dplyr), 
#as well as tm and MASS
library(tidyverse) 
library(tm)
library(MASS)

#Read in files:
train <- read_csv("train.csv") #Read in the comma separated value data file for training the model
test <- read_csv("test.csv") #Read in the csv data file for testing the model
sample <- read_csv("sample.csv") #Read in the csv data file for sample submission (for reference)


### PREPARE/CLEAN TRAINING SET ###

#Get the content
tweets = VCorpus(DataframeSource(train[,2]))

#Compute TF-IDF matrix and inspect sparsity
tweets.tfidf = DocumentTermMatrix(tweets, control = list(weighting = weightTfIdf))
tweets.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.

##### Reducing Term Sparsity #####

#Clean up the corpus
tweets.clean = tm_map(tweets, stripWhitespace)                          # remove extra whitespace
tweets.clean = tm_map(tweets.clean, removeNumbers)                      # remove numbers
tweets.clean = tm_map(tweets.clean, removePunctuation)                  # remove punctuation
tweets.clean = tm_map(tweets.clean, content_transformer(tolower))       # ignore case
tweets.clean = tm_map(tweets.clean, removeWords, stopwords("english"))  # remove stop words
tweets.clean = tm_map(tweets.clean, stemDocument)                       # stem all words

#Recompute TF-IDF matrix
tweets.clean.tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf))

#Inspect the first 5 documents and first 5 terms
tweets.clean.tfidf[1:5,1:5]
as.matrix(tweets.clean.tfidf[1:5,1:5])

#Remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.99 = removeSparseTerms(tweets.clean.tfidf, 0.99)  
tfidf.99
as.matrix(tfidf.99[1:5,1:5])
dtm.tfidf.99 = as.matrix(tfidf.99)

df.99.scored <- data.frame(dtm.tfidf.99)
df.99.scored["SCORE"] <- train[,1]


###### PREPARE TESTING SET ###### 

#Create a new test data frame with all colnames from training set so as to include variables 
#that might be used to create our models, but are not in the original test set
test2 <- rbind(test, c(1000, paste(colnames(df.99.scored), collapse = " ")))

#Get the content:
test2 = VCorpus(DataframeSource(test2[,2]))

#Compute TF-IDF matrix and inspect sparsity
test.tfidf = DocumentTermMatrix(test2, control = list(weighting = weightTfIdf))
test.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.

##### Reducing Term Sparsity #####

#Clean up the corpus
test.clean = tm_map(test2, stripWhitespace)                          # remove extra whitespace
test.clean = tm_map(test.clean, removeNumbers)                      # remove numbers
test.clean = tm_map(test.clean, removePunctuation)                  # remove punctuation
test.clean = tm_map(test.clean, content_transformer(tolower))       # ignore case
test.clean = tm_map(test.clean, removeWords, stopwords("english"))  # remove stop words
test.clean = tm_map(test.clean, stemDocument)                       # stem all words

#Recompute TF-IDF matrix and convert to dataframe
test.clean.tfidf = DocumentTermMatrix(test.clean, control = list(weighting = weightTfIdf))
test.clean.tfidf = as.matrix(test.clean.tfidf)
df.test.preds <- data.frame(test.clean.tfidf)
df.test.preds <- df.test.preds[-nrow(df.test.preds),]


### CREATE BASIC LINEAR MODEL WITH CLEANED 99% TRAINING SET ###

#Use all variables to create a linear model with the lm() function
lm.df.99 <- lm(SCORE~.,  
               data=df.99.scored)

summary(lm.df.99) #Look to see what variables have a significance greater than or equal to 0.1

lm.df.99.2 <- lm(SCORE~cant+dont+excit+googl+insur+less+need+pedal+safer+save+saw+
                   soon+thing+wait+want+warn+wrong, #Use all variables with any significance
                 data=df.99.scored)
summary(lm.df.99.2) #Recheck all variables and their new p-values

#Remove all variables with a p-value less than .05
#Try to run the model on the original test set -- if any words are not in the test set, also remove those
#NOTE -- this was done BEFORE creating a new test data frame with columns for each term in the training set
lm.df.99.3 <- lm(SCORE~cant+dont+excit+googl+insur+less+need+safer+
                   soon+thing+wait+want+warn+wrong, 
                 data=df.99.scored)   
summary(lm.df.99.3) #All variables still have p-value >= 0.05

#Residual standard error: 0.7346 on 966 degrees of freedom
#Multiple R-squared:  0.1393,	Adjusted R-squared:  0.1268 
#F-statistic: 11.17 on 14 and 966 DF,  p-value: < 2.2e-16


### BASIC LM PREDICTION... ###

#Use predict function to predict sentiment on test set using final basic linear model
mypreds <- data.frame(predict(lm.df.99.3, newdata = df.test.preds))

sentiment <- round(mypreds[,1],digits=0) #Round all values to closest integer
lm.preds <- as.data.frame(sentiment) #Place into a data frame

# Change <1 to 1 and >5 to 5 
lm.preds[lm.preds > 5,] = 5
lm.preds[lm.preds < 1,] = 1

#Add in ID numbers
lm.preds["id"] = test[,1]
lm.preds <- lm.preds[c(2,1)] #Switch columns, so they are in the correct order

write.table(lm.preds, file = "lm_car_tweets_eih.csv", row.names=F, sep=",") #Write out to a csv
#KAGGLE SCORE = 0.68302


### CREATE OPTIMAL LINEAR MODEL USING STEP FUNCTION ###

#Use step function to find optimal set of predictors

#step(lm.df.99, direction = "both") #THIS TAKES A WHILE TO RUN... RESULTS ARE SHOWN BELOW

lm.optimal <- lm(formula = SCORE ~ accid + cant + car + come + cool + dont + 
                   excit + fbi + googl + hit + insur + just + less + need + 
                   pedal + safer + save + saw + soon + taxi + thing + think + 
                   truck + wait + want + warn + wheel + will + wrong + yes, 
                 data = df.99.scored)

summary(lm.optimal)
#Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
#(Intercept)   3.1192     0.0527  59.186  < 2e-16 ***
#accid         0.5448     0.3633   1.499 0.134109    
#cant          0.8763     0.3397   2.580 0.010033 *  
#car          -5.5622     2.4863  -2.237 0.025511 *  
#come          0.4462     0.2374   1.880 0.060438 .  
#cool          0.4972     0.3367   1.476 0.140142    
#dont         -1.3466     0.2837  -4.746 2.39e-06 ***
#excit         2.1126     0.4150   5.091 4.29e-07 ***
#fbi          -0.7195     0.3708  -1.940 0.052624 .  
#googl         0.7496     0.2289   3.275 0.001096 ** 
#hit           0.4858     0.3457   1.405 0.160300    
#insur        -0.6228     0.2637  -2.362 0.018402 *  
#just         -0.4008     0.2523  -1.589 0.112464    
#less          0.6134     0.2688   2.282 0.022722 *  
#need          0.7889     0.2520   3.131 0.001795 ** 
#pedal         0.7273     0.4894   1.486 0.137549    
#safer         0.9553     0.4359   2.191 0.028663 *  
#save          0.8075     0.3342   2.416 0.015878 *  
#saw           0.5394     0.3240   1.665 0.096251 .  
#soon          0.5870     0.3000   1.957 0.050669 .  
#taxi         -0.4263     0.2958  -1.441 0.149795    
#thing         1.1907     0.3864   3.082 0.002116 ** 
#think        -0.3601     0.2513  -1.433 0.152153    
#truck        -0.6536     0.3527  -1.853 0.064202 .  
#wait          0.6692     0.2975   2.249 0.024727 *  
#want          0.8484     0.2176   3.899 0.000103 ***
#warn         -0.7596     0.3858  -1.969 0.049227 *  
#wheel        -0.6981     0.3977  -1.756 0.079493 .  
#will          0.4515     0.2315   1.950 0.051438 .  
#wrong        -0.7016     0.3152  -2.226 0.026237 *  
#yes           0.6362     0.3771   1.687 0.091866 .  

#Residual standard error: 0.7231 on 950 degrees of freedom
#Multiple R-squared:  0.1799,	Adjusted R-squared:  0.154 
#F-statistic: 6.946 on 30 and 950 DF,  p-value: < 2.2e-16


### OPTIMAL LM PREDICTION... ###

#Use predict function to make predictions on test set sentiment using optimal linear model
mypreds.optimal <- data.frame(predict(lm.optimal, newdata = df.test.preds))

sentiment <- round(mypreds.optimal[,1],digits=0) #Round all values to closest integer
lm.preds.optimal <- as.data.frame(sentiment) #Place into a data frame

# Change <1 to 1 and >5 to 5 
lm.preds.optimal[lm.preds.optimal > 5,] = 5
lm.preds.optimal[lm.preds.optimal < 1,] = 1

#Add in ID numbers
lm.preds.optimal["id"] = test[,1]
lm.preds.optimal <- lm.preds.optimal[c(2,1)] #Switch columns, so they are in the correct order

write.table(lm.preds.optimal, file = "lm_optimal_car_tweets.csv", row.names=F, sep=",") #Write out to a csv
#KAGGLE SCORE = ???


### USE LDA TO CREATE A NEW LINEAR MODEL ###

df.99.scored.2 <- df.99.scored[, !(colnames(df.99.scored) %in% c("univers"))]
z <- lda(SCORE ~ ., df.99.scored.2, CV = T)
sum(z$class == train$sentiment)/length(z$class)

### LDA PREDICTION ... ###

lm.preds.LDA["sentiment"] <- (predict(z, newdata=df.test.preds)$class)

#Add in ID numbers
lm.preds.LDA["id"] = test[,1]
lm.preds.LDA <- lm.preds.LDA[c(2,1)] #Switch columns, so they are in the correct order

write.table(lm.preds.LDA, file = "lm_optimal_car_tweets.csv", row.names=F, sep=",") #Write out to a csv
#KAGGLE SCORE = ???

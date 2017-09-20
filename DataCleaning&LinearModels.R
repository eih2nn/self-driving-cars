#TEAM COMPETITION 3-1
#Elizabeth Homan
#Ben Greenawald
#Leelakrishna (Sai) Bollempalli

#Load the core tidyverse packages (ggplot2, tibble, tidyr, readr, purrr, and dplyr), 
#as well as tm and MASS
library(tidyverse) 
library(tm)
library(MASS)
source("preprocess.R")

#Read in files:
train <- read_csv("train.csv") #Read in the comma separated value data file for training the model
test <- read_csv("test.csv") #Read in the csv data file for testing the model
sample <- read_csv("sample.csv") #Read in the csv data file for sample submission (for reference)


### PREPARE/CLEAN TRAINING SET ###
df.99.scored <- clean_data(train, 0.99, filter_symbol = F, stop_words = F, 
                            extra = T, weighting = "ntn", ngram = F)
corpus <- colnames(df.99.scored)[!(colnames(df.99.scored) %in% c("num_at", "num_exlaim",
                                                               "num_hash", "num_question", "odd_char"))]
df.99.scored["SCORE"] <- train[,1]


###### PREPARE TESTING SET ###### 
df.test.preds <- clean_data(test, 0.99, filter_symbol = F, stop_words = F, 
                            extra = F, dict = corpus, weighting = "ntn")

###############################################################
### CREATE BASIC LINEAR MODEL WITH CLEANED 99% TRAINING SET ###
###############################################################

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

#######################################################
### CREATE OPTIMAL LINEAR MODEL USING STEP FUNCTION ###
#######################################################

#Use step function to find optimal set of predictors

# step(lm.df.99, direction = "both") #THIS TAKES A WHILE TO RUN... RESULTS ARE SHOWN BELOW

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
#KAGGLE SCORE = 0.66 (strangely, not better)


### USE LDA TO CREATE A NEW LINEAR MODEL ###

df.99.scored.2 <- df.99.scored[, !(colnames(df.99.scored) %in% c("univers"))]
z <- lda(SCORE ~ ., df.99.scored.2, CV = T) #Assumes frequencies from training set
sum(z$class == train$sentiment)/length(z$class)

### LDA PREDICTION ... ###

lm.preds.LDA["sentiment"] <- (predict(z, newdata=df.test.preds)$class)

#Add in ID numbers
lm.preds.LDA["id"] = test[,1]
lm.preds.LDA <- lm.preds.LDA[c(2,1)] #Switch columns, so they are in the correct order

##########################################################
### CREATE BASIC LINEAR MODEL WITH ADDITIONAL FEATURES ###
##########################################################

### PREPARE/CLEAN TRAINING SET ###
df.99.scored2 <- clean_data(train, 0.99, filter_symbol = F, stop_words = F, 
                           extra = T, weighting = "ntn", ngram = F)
corpus2 <- colnames(df.99.scored2)[!(colnames(df.99.scored2) %in% c("num_at", "num_exlaim",
                                                                 "num_hash", "num_question", "odd_char"))]
df.99.scored2["SCORE"] <- train[,1]


###### PREPARE TESTING SET ###### 
df.test.preds2 <- clean_data(test, 0.99, filter_symbol = F, stop_words = F, 
                            extra = T, dict = corpus2, weighting = "ntn")


### MAKE THE MODEL ###
lm.df.99.2 <- lm(SCORE~., data=df.99.scored2)
summary(lm.df.99.2) #Look to see what variables have a significance greater than or equal to 0.1

### KEEP ONLY PREDICTIVE VARIABLES
lm.df.99.22 <- lm(SCORE ~ dont + excit + fbi + googl + insur + just + 
                    less + need + pedal + save + saw + thing + think +
                    truck + wait + want + warn + will + odd_char + 
                    num_exlaim + num_question, df.99.scored2)
summary(lm.df.99.22)

### REPEAT
lm.df.99.23 <- lm(SCORE ~ dont + excit + fbi + googl + insur + just + 
                    less + need + save + saw + thing + think +
                    truck + wait + want + warn + will + 
                    num_exlaim + num_question, df.99.scored2)
summary(lm.df.99.23)

#Use predict function to make predictions on test set sentiment using optimal linear model
mypreds.extra <- data.frame(predict(lm.df.99.23, newdata = df.test.preds2))

sentiment <- round(mypreds.extra[,1],digits=0) #Round all values to closest integer
lm.preds.extra <- as.data.frame(sentiment) #Place into a data frame

# Change <1 to 1 and >5 to 5 
lm.preds.extra[lm.preds.extra > 5,] = 5
lm.preds.extra[lm.preds.extra < 1,] = 1

#Add in ID numbers
lm.preds.extra["id"] = test[,1]
lm.preds.extra <- lm.preds.extra[c(2,1)] #Switch columns, so they are in the correct order

write.table(lm.preds.extra, file = "lm_extra_car_tweets.csv", row.names=F, sep=",") #Write out to a csv
#KAGGLE SCORE = ???

##############################################
### CREATE BASIC LINEAR MODEL WITH BIGRAMS ###
##############################################

### PREPARE/CLEAN TRAINING SET ###
df.99.scored3 <- clean_data(train, 0.995, filter_symbol = F, stop_words = F, 
                            extra = T, weighting = "ntn", ngram = T)
corpus3 <- colnames(df.99.scored3)[!(colnames(df.99.scored3) %in% c("num_at", "num_exlaim",
                                                                    "num_hash", "num_question", "odd_char"))]
df.99.scored3["SCORE"] <- train[,1]


###### PREPARE TESTING SET ###### 
df.test.preds3 <- clean_data(test, 0.995, filter_symbol = F, stop_words = F, 
                             extra = T, dict = corpus3, weighting = "ntn", ngram = T)


### MAKE THE MODEL ###
lm.df.99.3 <- lm(SCORE~., data=df.99.scored3)
summary(lm.df.99.3) #Look to see what variables have a significance greater than or equal to 0.1

summary(lm.df.99.3)$coefficients[summary(lm.df.99.3)$coefficients[ ,4] < 0.05, ] 

### KEEP ONLY PREDICTIVE VARIABLES
lm.df.99.32 <- lm(SCORE ~ amazon + arent + autom + awesom + best + `brake steer` + 
                    `cant wait` + `car cant` + `car hit` + `car use` + `car will` + 
                    caus + concern + design + dont + drive + `drive car` + excit + heard + 
                    hell + hope + insid + instead + less + lethal + love + `make car` +
                    mean + mobil + near + need + `one driverless` + peopl + play +
                    point + `possibl go` + potenti + program + put + `put driverless` + save + set + 
                    `steer wheel` + stori + street + sure + tell + thought + two + `use driverless` +
                    `wait driverless` + `want driverless` + warn + wheel + wrong + 
                    num_exlaim + num_question, data = df.99.scored3)
summary(lm.df.99.32)

### REPEAT
lm.df.99.33 <- lm(SCORE ~ arent  + awesom + best + 
                    `cant wait` + `car cant` + `car hit` + `car will` + 
                    concern + dont + `drive car` + excit + heard + 
                    hell + hope + instead + less + lethal + love +
                    mean + mobil + need + play +
                    point + potenti + program + put + `put driverless` + save + set + 
                    `steer wheel` + stori + tell + thought + two +
                    `wait driverless` + `want driverless` + warn + wheel + wrong + 
                    num_exlaim + num_question, data = df.99.scored3)
summary(lm.df.99.33)

### REPEAT
lm.df.99.34 <- lm(SCORE ~ arent  + awesom + best + 
                    `cant wait` + `car cant` + `car hit` + 
                    concern + dont + `drive car` + excit + heard + 
                    hell + hope + instead + less + lethal + love +
                    mean + mobil + need + play +
                    point + potenti + program + put + `put driverless` + save + set + 
                    stori + tell + thought + two +
                    `want driverless` + warn + wheel + wrong + 
                    num_exlaim + num_question, data = df.99.scored3)
summary(lm.df.99.34)

#Use predict function to make predictions on test set sentiment using optimal linear model
mypreds.bi <- data.frame(predict(lm.df.99.3, newdata = df.test.preds3))

sentiment <- round(mypreds.bi[,1],digits=0) #Round all values to closest integer
lm.preds.bi <- as.data.frame(sentiment) #Place into a data frame

# Change <1 to 1 and >5 to 5 
lm.preds.bi[lm.preds.bi > 5,] = 5
lm.preds.bi[lm.preds.bi < 1,] = 1

#Add in ID numbers
lm.preds.bi["id"] = test[,1]
lm.preds.bi <- lm.preds.extra[c(2,1)] #Switch columns, so they are in the correct order

write.table(lm.preds.extra, file = "lm_bi_car_tweets.csv", row.names=F, sep=",") #Write out to a csv
#KAGGLE SCORE = ???

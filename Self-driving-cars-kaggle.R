#KAGGLE TWITTER SELF DRIVING CARS COMPETITION
#Team = ***
#Elizabeth Homan, eih2nn


#####################################################

#Load the core tidyverse packages: ggplot2, tibble, tidyr, readr, purrr, and dplyr
library(tidyverse) 

#Read in files:
train <- read_csv("train.csv") #Read in the comma separated value data file for training the model
test <- read_csv("test.csv") #Read in the csv data file for testing the model
sample <- read_csv("sample.csv") #Read in the csv data file for sample submission (for reference)

#####################################################

##### DATA CLEANING AND PREPARATION #####


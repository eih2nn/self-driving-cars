# Twitter Sentiment Analysis
Kaggle In Class Competition re: Twitter Sentiment Analysis of Self-Driving Cars



### Group Roles

All group members participated in all parts of this project, but the primary responsibilities were distributed as follows:

* Elizabeth Homan: Github Coordinator and primary contributor to the linear analysis
* Sai Bollempalli: Exploratory data analysis and non-parametric analysis
* Ben Greenawald: Implemented the KNN algorithm and the preprocessing script



### Preprocessing

All of the preprocessing took place in a single R script that was then sourced in all other files. The principle function in the script was *clean_data* that took in raw data straight from the CSV and performed all necessary preprocessing, returning a data frame of features. This function takes in a variety of parameters which control how the data should be preprocessed. Some of these options include the option to remove oddly encoded symbols (probably emojis), the option to keep stop words in, the ability to specify how words should be weighted by the weightSMART function, and the ability to specify the sparcity threshold for our feature matrix. A further option affords the ability to add some additional features as described below. 



### Feature Extraction

TODO



### Parametric Approach

TODO



### Non-Parametric Approach

TODO



### Final Results

TODO



### Discussion

TODO
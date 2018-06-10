# install.packages("spFSR")
# install.packages("tidyverse")
# install.packages("e1071")

#install.packages('party')
library(party)
library(dplyr)
library(spFSR)
library(e1071)
library(parallel)
library(caret)

setwd("/home/tim/MLProject/")
getwd()

ks <- read.csv("Kickstarter_filtered.csv")

ks <- ks %>% select(-matches("X")) #drop the index (x) variable

#taking random sample to ensure code runs
set.seed(12)
ks_samp<-ks[sample(nrow(ks), 40000), ]

#log transforming all exponentially distributed continuous variables
ks_samp$usd_pledged_real<-log(ks_samp$usd_pledged_real+.1)
ks_samp$usd_goal_real<-log(ks_samp$usd_goal_real+.1)
ks_samp$backers<-log(ks_samp$backers+.1)

#set the target and descriptive features
## TODO: Need to convert these to integers for knn classification
target <- ks_samp %>% pull(state)
descriptive <- ks_samp %>% select(-state)

dt_task <- makeClassifTask(data = cbind(target, descriptive), target = 'target')

dt_measure <- mmce

dt_wrapper <- makeLearner('classif.ctree')

dt_rdesc <- makeResampleDesc("RepCV", folds=5, reps=3)
#TODO Visualise this output for the report

dt_repcrosval <- resample(dt_wrapper
                          , dt_task
                          , dt_rdesc
                          , measures=dt_measure)


bayes_mean_result <- mean(bayes_repcrosval$measures.test[[2]])
cat('Repeated CV error with full set of features =', 100 * round(bayes_mean_result, 3))

set.seed(8, kind = "L'Ecuyer-CMRG")
set.seed(9, kind = "L'Ecuyer-CMRG")

dt_spsaMod <- spFeatureSelection(task = dt_task,
                                 wrapper = dt_wrapper, 
                                 num.features.selected = 0, ## set to 0 for automatic feature selection
                                 measure = dt_measure)



summary(dt_spsaMod)

dt_spsaMod$features

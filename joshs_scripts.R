library(readr)
library(dplyr)
library(mlbench)
library(tidyverse)
library(spFSR)
library(mlr)
ks <- read_csv("Kickstarter_filtered.csv")
View(ks)

#removing index and converting char to factors
ks$X1<-NULL
ks[sapply(ks, is.character)] <- lapply(ks[sapply(ks, is.character)], as.factor)

#log transforming all exponentially distributed continuous variables
ks$usd_pledged_real<-log(ks$usd_pledged_real+.1)
ks$usd_goal_real<-log(ks$usd_goal_real+.1)
ks$backers<-log(ks$backers+.1)

#normalising all numerical features
normalise<-function(x){
  x <- (x-min(x))/(max(x)-min(x))
}

ks[sapply(ks, is.numeric)]  <- lapply(ks[sapply(ks, is.numeric)], normalise)

#converting char features to factors

ks[sapply(ks, is.character)]  <- lapply(ks[sapply(ks, is.character)], factor)

### CONVERT DUMMIES TO THE FACTOR ###
library(caret)
ks_knn<-ks

#doesnt work?###########################################
factors<-colnames(ks[sapply(ks, is.factor)])
header<-vector()

for (i in 1:length(factors)){
  paste("dummies_",i,sep="") <- predict(dummyVars(~ factors[i], data = ks), newdata = ks_knn)
  header[i] <- unlist(strsplit(colnames(paste("dummies_",i,sep="")), '[.]'))[2 * (1:ncol(paste("dummies_",i,sep="")))]
  factors[i] <- factor(paste("dummies_",i,sep="") %*% 1:ncol(paste("dummies_",i,sep="")), labels = header[i])
}
#####################################################3


dummies_cat <- predict(dummyVars(~ main_category, data = ks), newdata = ks_knn)
header <- unlist(strsplit(colnames(dummies_cat), '[.]'))[2 * (1:ncol(dummies_cat))]
main_category <- factor(dummies_cat %*% 1:ncol(dummies_cat), labels = header)
ks_knn$main_category<-main_category

dummies_state <- predict(dummyVars(~ state, data = ks), newdata = ks_knn)
header <- unlist(strsplit(colnames(dummies_state), '[.]'))[2 * (1:ncol(dummies_state))]
state <- factor(dummies_state %*% 1:ncol(dummies_state), labels = header)
ks_knn$state<-state

dummies_cont <- predict(dummyVars(~ continent, data = ks), newdata = ks_knn)
header <- unlist(strsplit(colnames(dummies_cont), '[.]'))[2 * (1:ncol(dummies_cont))]
continent <- factor(dummies_cont %*% 1:ncol(dummies_cont), labels = header)
ks_knn$continent<-continent


classif.task<- makeClassifTask(id = 'ks',data=ks_knn,target="state")

learners <- makeLearners(c('randomForest','rpart','kknn'),
                         type = "classif", predict.type = "response")

kknn.model<-mlr::train(learners$classif.kknn,classif.task)

#works :D

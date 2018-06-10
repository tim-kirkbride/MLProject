load_libraries <- function(){
  library(readr)
  library(dplyr)
  library(mlbench)
  #library(tidyverse)
  library(mlr)
  library(kknn)
  library(caret)
}

import_data <- function(){
  ks <- read_csv("Kickstarter_filtered.csv")
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
  
  #taking random sample to ensure code runs
  set.seed(12)
  ks_samp<-ks[sample(nrow(ks), 40000), ]
  
  prop.table(table(ks_samp$state))
  
  #partitioning training/validation : test (70:30)
  train.index <- createDataPartition(ks_samp$state, p = .7, list = FALSE)
  train <- ks_samp[ train.index,]
  test  <- ks_samp[-train.index,]
  
  prop.table(table(train$state))
  prop.table(table(test$state))
}

makeModelAndPredict <- function(classif_name){
  
  #template 
  task <- makeClassifTask(data = train, target = 'state', id = 'ks')
  #classif_name <- "classif.naiveBayes"
  classif_id <- switch(
    classif_name 
    ,"classif.kknn" = "knn"
    ,"classif.rpart" = "rpart"
    ,"classif.randomForest" = "randomForest"
    ,"classif.naiveBayes" = "NB" 
      )
  print(classif_id)
  learner <- makeLearner(classif_name, id= classif_id, predict.type = 'prob')
  
  #parameters have already been tuned
  rds_name <- switch(
    classif_name 
    ,"classif.kknn" = "res_knn2.rds"
    ,"classif.rpart" = "res_rpart.rds"
    ,"classif.randomForest" = "res_rf.rds"
    ,"classif.naiveBayes" = "res_nb.rds" 
  )
  res <- readRDS(rds_name)
  res$x
  
  #fusing optimal parameters into tuned learners
  tunedLrn <- setHyperPars(makeLearner((classif_name) ,predict.type = 'prob'), par.vals = res$x)
  print(tunedLrn)
  
  #training tuned learners
  tunedMod <- mlr::train(tunedLrn, task) 
  print(tunedMod)
  
  #predict on training data
  tunedPred<- predict(tunedMod, task)
  print(tunedPred)
  #Threshold analysis of training prediction
  d <- generateThreshVsPerfData(tunedPred, measures = list(mmce))
  plotThreshVsPerf(d) + labs(title = 'Threshold Adjustment', x = 'Threshold')
  threshold <- mean(d$data[d$data$mmce==min(d$data$mmce),]$threshold) 
  print(threshold)
  
  #predict on test data 
  testPred <- predict(tunedMod, newdata = test)
  print(testPred)
  performance(testPred)
  
  #predict on test with threshold 
  threshold_testPred <- setThreshold(testPred, threshold)
  print(threshold_testPred)
  
  print(calculateConfusionMatrix( threshold_testPred,relative = TRUE))
  performance( threshold_testPred )
  #include more performance metrics
}

load_libraries()
import_data()

makeModelAndPredict("classif.kknn")
makeModelAndPredict("classif.naiveBayes")
makeModelAndPredict("classif.randomForest")
makeModelAndPredict("classif.rpart")

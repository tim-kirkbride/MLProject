library(readr)
library(dplyr)
library(mlbench)
library(tidyverse)
library(mlr)
library(kknn)
library(caret)


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

#template
task <- makeClassifTask(data = train, target = 'state', id = 'ks')

# Configure learners with probability type
lrns <- list(
  makeLearner("classif.naiveBayes", id = "NB", predict.type = 'prob'),
  makeLearner("classif.rpart", id = "rpart", predict.type = 'prob'),
  makeLearner("classif.randomForest", id = "randomForest", predict.type = 'prob'),
  makeLearner('classif.kknn', id= "knn", predict.type = 'prob')
)

#feature selection

mFV <- generateFilterValuesData(task,
                                method = c('information.gain',
                                           'chi.squared'))
plotFilterValues(mFV)

#number of features tuning
ps.feat <- makeParamSet(makeDiscreteParam("fw.abs", values = c(2,3,4)))
rdesc.feat <- makeResampleDesc("CV", iters = 3)

#nb
lrn.nb <- makeFilterWrapper(learner = lrns[[1]], fw.method = "chi.squared")

res.feat.nb <- tuneParams(lrn.nb, task =task,
                             resampling = rdesc.feat,
                             par.set = ps.feat,
                             control = makeTuneControlGrid())

fusedLrn.nb <- makeFilterWrapper(learner = "classif.naiveBayes",
                                    fw.method = "chi.squared",
                                    fw.abs = res.feat.nb$x$fw.abs)
#optimal is 4 features

#rpart
lrn.rpart <- makeFilterWrapper(learner = lrns[[2]], fw.method = "chi.squared")

res.feat.rpart <- tuneParams(lrn.rpart, task =task,
                  resampling = rdesc.feat,
                  par.set = ps.feat,
                  control = makeTuneControlGrid())

fusedLrn.rpart <- makeFilterWrapper(learner = "classif.rpart",
                              fw.method = "chi.squared",
                              fw.abs = res.feat.rpart$x$fw.abs)

#optimal is 3 features

#rf
lrn.rf <- makeFilterWrapper(learner = lrns[[3]], fw.method = "chi.squared")

res.feat.rf <- tuneParams(lrn.rf, task =task,
                             resampling = rdesc.feat,
                             par.set = ps.feat,
                             control = makeTuneControlGrid())

fusedLrn.rf <- makeFilterWrapper(learner = "classif.randomForest",
                              fw.method = "chi.squared",
                              fw.abs = res.feat.rf$x$fw.abs)

#optimal is 3

#knn
lrn.knn <- makeFilterWrapper(learner = lrns[[4]], fw.method = "chi.squared")

res.feat.knn <- tuneParams(lrn.knn, task =task,
                          resampling = rdesc.feat,
                          par.set = ps.feat,
                          control = makeTuneControlGrid())

fusedLrn.knn <- makeFilterWrapper(learner = "classif.kknn",
                                 fw.method = "chi.squared",
                                 fw.abs = res.feat.knn$x$fw.abs)

#optimal 3

#####Initial tuning searches

# For naiveBayes, we can fine-tune Laplacian
ps.nb <- makeParamSet(
  makeDiscreteParam('laplace', values=c(0,10,25,50,100,200))
)

# For decision tree, fine-tune min.split i.e. minimum number of observations in a node for a split to be attempted
ps.rpart <- makeParamSet(
  makeDiscreteParam('minsplit', values=c(5,150,300,450,600)),
  makeDiscreteParam("minbucket", values=c(5,100,200,300))
)


# For randomForest, fine-tune mtry i.e mumber of variables and ntree i.e number of trees 
# sampled as candidates at each split. Seeing as there are only 6 descriptive variables, smaller values will be considered
ps.rf <- makeParamSet(
  makeDiscreteParam('mtry', values = c(1,2,3)),
  makeDiscreteParam("ntree", values = c(10,20,30))
)

# For kknn, we fine-tune k for values between 1-30
ps.knn <- makeParamSet(
  makeDiscreteParam('k', values=c(1,3,5,10,20,30))
)


# Configure tune control grid search and a 5-CV stratified sampling

ctrl.grd <- makeTuneControlGrid()
rdesc <- makeResampleDesc("CV", iters = 5, stratify = TRUE)

#parameter tuning

#nb
res.nb = tuneParams(fusedLrn.nb, task = task, control = ctrl.grd,
                 measures = mmce, resampling = rdesc,
                 par.set = ps.nb)

#rpart
res.rpart = tuneParams(fusedLrn.rpart, task = task, control = ctrl.grd,
                       measures = mmce, resampling = rdesc,
                       par.set = ps.rpart)

#rf
res.rf = tuneParams(fusedLrn.rf, task = task, control = ctrl.grd,
                    measures = mmce, resampling = rdesc,
                    par.set = ps.rf)

#knn
res.knn = tuneParams(fusedLrn.knn, task = task, control = ctrl.grd,
                     measures = mmce, resampling = rdesc,
                     par.set = ps.knn)

#dont run again #############################
saveRDS(res.nb, file = "res_nb.rds")
saveRDS(res.rpart, file = "res_rpart.rds")
saveRDS(res.rf, file = "res_rf.rds")
#############################################

res.nb<-readRDS(file = "res_nb.rds")


#parameter plots

#nb
ps.nb.plot <- generateHyperParsEffectData(res.nb)
plotHyperParsEffect(ps.nb.plot, x = "laplace", y = "mmce.test.mean", plot.type = "line")

res.nb$x
#0 is clearly the optimal value of the laplace smoothing parameter, suggesting no smoothing is required

#rpart
ps.rpart.plot = generateHyperParsEffectData(res.rpart)
plotHyperParsEffect(ps.rpart.plot, x = "minsplit", y = "minbucket", z = "mmce.test.mean",
                    plot.type = "heatmap")+
  scale_x_continuous(breaks=c(5,150,300,450,600))+
  scale_y_continuous(breaks=c(5,100,200,300))

res.rpart$x
# 5 is shown to be the optimal value for both the minsplit and minbucket parameters
# the plot suggests that these specfic value are not particularly notable, as similar
# mmce measures were found for minsplit values up to 450 and minbucket values up to 200
# given very little variance in the mmce measure is seen across these broad ranges, 
# further tuning was deemed unnecessary and was not attempted

#rf
ps.rf.plot <- generateHyperParsEffectData(res.rf)
plotHyperParsEffect(ps.rf.plot, x = "mtry", y = "ntree", z = "mmce.test.mean",
                    plot.type = "heatmap")+scale_y_continuous(breaks=c(10,20,30))

res.rf$x

#the plot clearly shows that the top corner cell, corresponding to mtry = 3 and 
#ntree = 30 had the lowest mmce of all other combinations tested

#knn
ps.knn.plot <- generateHyperParsEffectData(res.knn)
plotHyperParsEffect(ps.knn.plot, x = "k", y = "mmce.test.mean", plot.type = "line")

res.knn$x

#here the optimal k value of 10 is shown as the lowest point in the plot
#the plot shows that there is an initial dip in the mmce for k values between 3 and 20
#and as such, a more appropriate k value may exist. further tuning was therefore attempted

#knn retuning 
ps.knn2 <- makeParamSet(
  makeDiscreteParam('k', values=seq(4,18,by=2))
)

res.knn2 = tuneParams(fusedLrn.knn, task = task, control = ctrl.grd,
                     measures = mmce, resampling = rdesc,
                     par.set = ps.knn2)

###### dont run again ################
saveRDS(res.knn2, file = "res_knn2.rds")
#######################################

ps.knn2.plot <- generateHyperParsEffectData(res.knn2)
plotHyperParsEffect(ps.knn2.plot, x = "k", y = "mmce.test.mean", plot.type = "line")

res.knn2$x

#the variation in mmce with k is now much clearer for the range 4 - 18, and it can be 
#seen that k = 4 is the optimal point for minimising mmce.



#fusing optimal parameters into tuned learners
#knn
tunedLrn.knn <- setHyperPars(makeLearner(("classif.kknn")), par.vals = res.knn2$x) 

#training tuned learners
tunedMod.knn <- mlr::train(tunedLrn.knn, task) 

#predicting using tuned models (dont do yet)
tunedPred.knn <- predict(tunedMod.knn, newdata = test)

#threshold adjustment

###############################################

#construct wrappers with tuning constraints
tunedlrn.nb <- makeTuneWrapper(lrns[[1]], rdesc, mmce, ps.nb, ctrl.grd)
tunedlrn.rpart <- makeTuneWrapper(lrns[[2]], rdesc, mmce, ps.rpart, ctrl.grd)
tunedlrn.rf <- makeTuneWrapper(lrns[[3]], rdesc, mmce, ps.rf, ctrl.grd)
tunedlrn.knn <- makeTuneWrapper(lrns[[4]], rdesc, mmce, ps.knn, ctrl.grd)

#train models with tuned learners 
tunedMod.nb  <- mlr::train(tunedlrn.nb, task)
tunedMod.rpart  <- mlr::train(tunedlrn.rpart, task)
tunedMod.rf <- mlr::train(tunedlrn.rf, task)
tunedMod.knn  <- mlr::train(tunedlrn.knn, task)

#########################################

#predicting on training
tunedPred.nb <- predict(tunedMod.nb, task)
tunedPred.rpart <- predict(tunedMod.rpart, task)
tunedPred.rf <- predict(tunedMod.rf, task)
tunedPred.knn<- predict(tunedMod.knn, task)

#threshold assessment
d1 <- generateThreshVsPerfData(tunedPred.nb, measures = list(mmce))
d2 <- generateThreshVsPerfData(tunedPred.rpart, measures = list(mmce))
d3 <- generateThreshVsPerfData(tunedPred.rf, measures = list(mmce))
d4 <- generateThreshVsPerfData(tunedPred.knn, measures = list(mmce))

#plotting thresholds
plotThreshVsPerf(d1) + labs(title = 'Threshold Adjustment for Naive Bayes', x = 'Threshold')
plotThreshVsPerf(d2) + labs(title = 'Threshold Adjustment for Decision Tree', x = 'Threshold')
plotThreshVsPerf(d3) + labs(title = 'Threshold Adjustment for Random Forest', x = 'Threshold')
plotThreshVsPerf(d4) + labs(title = 'Threshold Adjustment for ...-KNN', x = 'Threshold')

#to find ideal threshold point (maybe just report range instead of mean?)
mean(d1$data[d1$data$mmce==min(d1$data$mmce),]$threshold)
mean(d2$data[d2$data$mmce==min(d2$data$mmce),]$threshold)
mean(d3$data[d3$data$mmce==min(d3$data$mmce),]$threshold)
mean(d4$data[d4$data$mmce==min(d4$data$mmce),]$threshold)





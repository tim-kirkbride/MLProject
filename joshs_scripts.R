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
plotFilterValues(mFV)+ggtitle("Figure 1. Feature Importance Assessment")

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
res.feat.nb$y
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

res.feat.rpart$y
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

res.feat.rf$y
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

res.feat.knn$y
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
res.nb$y
#rpart
res.rpart = tuneParams(fusedLrn.rpart, task = task, control = ctrl.grd,
                       measures = mmce, resampling = rdesc,
                       par.set = ps.rpart)
res.rpart$x
res.rpart$y
#rf
res.rf = tuneParams(fusedLrn.rf, task = task, control = ctrl.grd,
                    measures = mmce, resampling = rdesc,
                    par.set = ps.rf)

res.rf$x
res.rf$y
#knn
res.knn = tuneParams(fusedLrn.knn, task = task, control = ctrl.grd,
                     measures = mmce, resampling = rdesc,
                     par.set = ps.knn)

#dont run again #############################
saveRDS(res.nb, file = "res_nb.rds")
saveRDS(res.rpart, file = "res_rpart.rds")
saveRDS(res.rf, file = "res_rf.rds")
#############################################



#parameter plots

#nb
ps.nb.plot <- generateHyperParsEffectData(res.nb)
plotHyperParsEffect(ps.nb.plot, x = "laplace", y = "mmce.test.mean", plot.type = "line")+
  ggtitle("Figure 2. Line Plot of Naive Bayes Tuning Interations")

res.nb$x
#0 is clearly the optimal value of the laplace smoothing parameter, suggesting no smoothing is required

#rpart
ps.rpart.plot = generateHyperParsEffectData(res.rpart)
plotHyperParsEffect(ps.rpart.plot, x = "minsplit", y = "minbucket", z = "mmce.test.mean",
                    plot.type = "heatmap")+
  scale_x_continuous(breaks=c(5,150,300,450,600))+
  scale_y_continuous(breaks=c(5,100,200,300))+ggtitle("Figure 3. Heatmap of Decision Tree Tuning Interations")

res.rpart$x
# 5 is shown to be the optimal value for both the minsplit and minbucket parameters
# the plot suggests that these specfic value are not particularly notable, as similar
# mmce measures were found for minsplit values up to 450 and minbucket values up to 200
# given very little variance in the mmce measure is seen across these broad ranges, 
# further tuning was deemed unnecessary and was not attempted

#rf
ps.rf.plot <- generateHyperParsEffectData(res.rf)
plotHyperParsEffect(ps.rf.plot, x = "mtry", y = "ntree", z = "mmce.test.mean",
                    plot.type = "heatmap")+scale_y_continuous(breaks=c(10,20,30))+
  ggtitle("Figure 4. Heatmap of Random Forest Tuning Interations")

res.rf$x

#the plot clearly shows that the top corner cell, corresponding to mtry = 3 and 
#ntree = 30 had the lowest mmce of all other combinations tested

#knn
ps.knn.plot <- generateHyperParsEffectData(res.knn)
plotHyperParsEffect(ps.knn.plot, x = "k", y = "mmce.test.mean", plot.type = "line")+
  ggtitle("Figure 5. Line Plot of KNN Tuning Interations")


#here the optimal k value of 10 is shown as the lowest point in the plot
#the plot shows that there is an initial dip in the mmce for k values between 3 and 20
#and as such, a more appropriate k value may exist. further tuning was therefore attempted

#knn retuning 
ps.knn2 <- makeParamSet(
  makeDiscreteParam('k', values=seq(1,8,by=1))
)

res.knn2 = tuneParams(fusedLrn.knn, task = task, control = ctrl.grd,
                     measures = mmce, resampling = rdesc,
                     par.set = ps.knn2)

###### dont run again ################
saveRDS(res.knn2, file = "res_knn2.rds")
#######################################

ps.knn2.plot <- generateHyperParsEffectData(res.knn2)
plotHyperParsEffect(ps.knn2.plot, x = "k", y = "mmce.test.mean", plot.type = "line")+
  ggtitle("Figure 6. Line Plot of Specialised KNN Tuning Interations")

res.knn2$x
res.knn2$y
#the variation in mmce with k is now much clearer for the range 4 - 18, and it can be 
#seen that k = 4 is the optimal point for minimising mmce.


#fusing optimal parameters into tuned learners

tunedLrn.nb <- setHyperPars(makeLearner(("classif.naiveBayes") ,predict.type = 'prob'), par.vals = res.nb$x)
tunedLrn.rpart <- setHyperPars(makeLearner(("classif.rpart") ,predict.type = 'prob'), par.vals = res.rpart$x)
tunedLrn.rf <- setHyperPars(makeLearner(("classif.randomForest") ,predict.type = 'prob'), par.vals = res.rf$x)
tunedLrn.knn <- setHyperPars(makeLearner(("classif.kknn") ,predict.type = 'prob'), par.vals = res.knn2$x)

#defining models
mod.nb<-mlr::train(tunedLrn.nb, task = task)
mod.rpart<-mlr::train(tunedLrn.rpart, task = task)
mod.rf<-mlr::train(tunedLrn.rf, task = task)
mod.knn<-mlr::train(tunedLrn.knn, task = task)

#prediction task
test.task <- makeClassifTask(id = "test",
                                data = test,
                                target = "state")
#prediction on test
pred.nb<- predict(mod.nb,test.task)
pred.rpart <- predict(mod.rpart,test.task)
pred.rf <- predict(mod.rf,test.task)
pred.knn <- predict(mod.knn,test.task)

#threshold adjustment
d.nb  <- generateThreshVsPerfData(pred.nb, measures = mmce)
d.rpart<- generateThreshVsPerfData(pred.rpart, measures = mmce)
d.rf<- generateThreshVsPerfData(pred.rf, measures = mmce)
d.knn<- generateThreshVsPerfData(pred.knn, measures = mmce)

#threshold plots
plotThreshVsPerf(d.nb) + labs(title = 'Figure 7. Threshold Adjustment for Naive Bayes', x = 'Threshold')
plotThreshVsPerf(d.rpart) + labs(title = 'Figure 8. Threshold Adjustment for Decision Tree', x = 'Threshold')
plotThreshVsPerf(d.rf) + labs(title = 'Figure 9. Threshold Adjustment for Random Forest', x = 'Threshold')
plotThreshVsPerf(d.knn) + labs(title = 'Figure 10. Threshold Adjustment for 2-KNN', x = 'Threshold')

#to find ideal threshold point (maybe just report range instead of mean?)

nb.thresh<-mean(d.nb$data[d.nb$data$mmce==min(d.nb$data$mmce),]$threshold)
rpart.thresh<-mean(d.rpart$data[d.rpart$data$mmce==min(d.rpart$data$mmce),]$threshold)
rf.thresh<-mean(d.rf$data[d.rf$data$mmce==min(d.rf$data$mmce),]$threshold)
knn.thresh<-mean(d.knn$data[d.knn$data$mmce==min(d.knn$data$mmce),]$threshold)


threshPred.nb <- setThreshold(pred.nb,nb.thresh)
threshPred.rpart <- setThreshold(pred.rpart,rpart.thresh)
threshPred.rf <- setThreshold(pred.rf,rf.thresh)
threshPred.knn <-setThreshold(pred.knn,rf.thresh)

performance(threshPred.nb)
performance(threshPred.rpart)
performance(threshPred.rf)
performance(threshPred.knn)


#training tuned learners
tunedMod.rpart <- mlr::train(tunedLrn.rpart, task) 

#predict on training data
tunedPred<- predict(tunedMod, task)
print(tunedPred)
#Threshold analysis of training prediction
d <- generateThreshVsPerfData(tunedPred, measures = list(mmce))
plotThreshVsPerf(d) + labs(title = 'Threshold Adjustment', x = 'Threshold')
threshold <- mean(d$data[d$data$mmce==min(d$data$mmce),]$threshold) 
print(threshold)
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
d.nb <- generateThreshVsPerfData(tunedPred.nb, measures = list(mmce))
d.rpart <- generateThreshVsPerfData(tunedPred.rpart, measures = list(mmce))
d.rf <- generateThreshVsPerfData(tunedPred.rf, measures = list(mmce))
d.knn <- generateThreshVsPerfData(tunedPred.knn, measures = list(mmce))

#plotting thresholds
plotThreshVsPerf(d.nb) + labs(title = 'Threshold Adjustment for Naive Bayes', x = 'Threshold')
plotThreshVsPerf(d.rpart) + labs(title = 'Threshold Adjustment for Decision Tree', x = 'Threshold')
plotThreshVsPerf(d.rf) + labs(title = 'Threshold Adjustment for Random Forest', x = 'Threshold')
plotThreshVsPerf(d.knn) + labs(title = 'Threshold Adjustment for 4-KNN', x = 'Threshold')

#to find ideal threshold point (maybe just report range instead of mean?)


calculateConfusionMatrix(pred.nb)
performance(pred.nb)

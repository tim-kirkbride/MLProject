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

#partitioning training/validation : test (70:30)
train.index <- createDataPartition(ks_samp$state, p = .7, list = FALSE)
train <- ks_samp[ train.index,]
test  <- ks_samp[-train.index,]


#template
task <- makeClassifTask(data = train, target = 'state', id = 'ks')

# Configure learners with probability type
lrns <- list(
  makeLearner("classif.naiveBayes", id = "NB", predict.type = 'prob'),
  makeLearner("classif.rpart", id = "rpart", predict.type = 'prob'),
  makeLearner("classif.randomForest", id = "randomForest", predict.type = 'prob'),
  makeLearner('classif.kknn', id= "knn", predict.type = 'prob')
)

#####Initial tuning searches

# For naiveBayes, we can fine-tune Laplacian
ps.nb <- makeParamSet(
  makeDiscreteParam('laplace', values=c(0,10,25,50,100,200,500))
)

# For decision tree, fine-tune min.split i.e. minimum number of observations in a node for a split to be attempted
ps.rpart <- makeParamSet(
  makeDiscreteParam('minsplit', values=c(5,150,295,440,585)),
  makeDiscreteParam("minbucket", values=c(5,105,205,305))
)


# For randomForest, fine-tune mtry i.e mumber of variables randomly 
# sampled as candidates at each split. Seeing as there are only 6 descriptive variables, smaller values will be considered
ps.rf <- makeParamSet(
  makeDiscreteParam('mtry', values = c(1,2,3)),
  makeDiscreteParam("ntree", values = c(10,30))
)

# For kknn, we fine-tune k for values between 1-200
ps.knn <- makeParamSet(
  makeDiscreteParam('k', values=c(1,2,10,50,100,200))
)


# Configure tune control grid search and a 5-CV stratified sampling

ctrl.grd <- makeTuneControlGrid()
rdesc <- makeResampleDesc("CV", iters = 5, stratify = TRUE)

#assessing parameters

#nb
res.nb = tuneParams(lrns[[1]], task = task, control = ctrl.grd,
                 measures = mmce, resampling = rdesc,
                 par.set = ps.nb)
#plot
ps.nb.plot <- generateHyperParsEffectData(res.nb)
plotHyperParsEffect(ps.nb.plot, x = "laplace", y = "mmce.test.mean", plot.type = "line")

#rpart
res.rpart = tuneParams(lrns[[2]], task = task, control = ctrl.grd,
                    measures = mmce, resampling = rdesc,
                    par.set = ps.rpart)

#plot
ps.rpart.plot = generateHyperParsEffectData(res.rpart)
plotHyperParsEffect(ps.rpart.plot, x = "minsplit", y = "minbucket", z = "mmce.test.mean",
                    plot.type = "heatmap")

#rf
res.rf = tuneParams(lrns[[3]], task = task, control = ctrl.grd,
                       measures = mmce, resampling = rdesc,
                       par.set = ps.rf)
#plot
ps.rf.plot <- generateHyperParsEffectData(res.rf)
plotHyperParsEffect(ps.rf.plot, x = "mtry", y = "mmce.test.mean", plot.type = "line")

#

#construct wrappers with with tuning constraints
tunedlrn.nb <- makeTuneWrapper(lrns[[1]], rdesc, mmce, ps.nb, ctrl.grd)
tunedlrn.rpart <- makeTuneWrapper(lrns[[2]], rdesc, mmce, ps.rpart, ctrl.grd)
tunedlrn.rf <- makeTuneWrapper(lrns[[3]], rdesc, mmce, ps.rf, ctrl.grd)
tunedlrn.knn <- makeTuneWrapper(lrns[[4]], rdesc, mmce, ps.knn, ctrl.grd)

#train models with tuned learners 
tunedMod.nb  <- mlr::train(tunedlrn.nb, task)
tunedMod.rpart  <- mlr::train(tunedlrn.rpart, task)
tunedMod.rf <- mlr::train(tunedlrn.rf, task)
tunedMod.knn  <- mlr::train(tunedlrn.knn, task)

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

#####################################################################################







#classification task
classif.task <- makeClassifTask(id = "ks",
                                data = train,
                                target = "state")

#learners
lrns <- list(
  makeLearner("classif.naiveBayes", id = "NB", predict.type = 'prob'),
  makeLearner("classif.rpart", id = "rpart", predict.type = 'prob'),
  makeLearner("classif.randomForest", id = "randomForest", predict.type = 'prob'),
  makeLearner('classif.kknn', predict.type = 'prob')
)

#param tuning

#naive bayes
ps.nb <- makeParamSet(
  makeNumericParam('laplace',lower=0,upper=100)
)

ctrl.nb <- makeTuneControlRandom(maxit = 10) 
rdesc <- makeResampleDesc("CV", iters=5, stratify = TRUE) 

res.nb <- tuneParams(lrns[[1]], 
                  task = classif.task, 
                  resampling = rdesc, 
                  par.set = ps.nb, 
                  control = ctrl.nb)

#all interations show the same mmce, therefore smoothing parameter has no effect/is redundant

#rpart
ps.rp<-makeParamSet(
  makeIntegerParam("minsplit", lower = 2, upper = 100)
)

ctrl.rp <- makeTuneControlRandom(maxit = 3) 
rdesc <- makeResampleDesc("CV", stratify = TRUE) 

res.rp <- tuneParams(lrns[[2]], 
                     task = classif.task, 
                     resampling = rdesc, 
                     par.set = ps.rp, 
                     control = ctrl.rp)
#again all iterations have the same mmce

#random forest
ps.rf <- makeParamSet(
  makeIntegerParam('mtry',lower=2,upper=4)
)

ctrl.rf <- makeTuneControlGrid() 
rdesc <- makeResampleDesc("CV", iters=5, stratify.cols = "state") 

set.seed(12)
res.rf <- tuneParams(lrns[[3]], 
                     task = classif.task, 
                     resampling = rdesc, 
                     par.set = ps.rf, 
                     control = ctrl.rf)

#knn
ps.knn<-makeParamSet(
  makeDiscreteParam('k', values = seq(2, 20, by = 1))
)

ctrl.knn <- makeTuneControlGrid() 
rdesc <- makeResampleDesc("CV", iters=5L, stratify.cols = "state") 

res.rp <- tuneParams(lrns[[4]], 
                     task = classif.task, 
                     resampling = rdesc, 
                     par.set = ps.knn, 
                     control = ctrl.knn)


# Configure tune wrapper with tune-tuning settings (knn)
tunedLearner1 <- makeTuneWrapper(lrns[[4]], rdesc, mmce, ps.knn, ctrl)


#feature selection (mlr)
library(rJava)
library(FSelector)
library(randomForestSRC)

classif.task<-makeClassifTask(id="ks",data=ks,target="state")

fv<-generateFilterValuesData(classif.task,method = 'chi.squared')

plotFilterValues(fv) + coord_flip()

lrn <- makeFilterWrapper(learner = "classif.kknn", fw.method = "chi.squared")

ps <- makeParamSet(makeDiscreteParam("fw.perc", values = seq(0.1, 0.4, 0.05)))
rdesc <- makeResampleDesc("CV", iters = 3)
res <- tuneParams(lrn, task = classif.task,
                  resampling = rdesc,
                  par.set = ps,
                  control = makeTuneControlGrid())

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

kknn.model<-mlr::train(lrns[[4]],classif.task)

#works :D

classif.task<- makeClassifTask(id = 'ks',data=ks,target="state")


#feature selection
## last column is the target variable Y
Y <- ks %>% pull(state)

## other columns make up the feature matrix
X <- ks %>% select(-state)

## set the MLR classification task
sel.task <- makeClassifTask(data = cbind(train,test, target = state))

## set the performance measure
sel.measure <- mmce ## mean misclassification error

## set the wrapper classification algorithm
#sel.knn.wrapper <- makeLearner("classif.knn", k = 1)

## you can try other algorithms as well
sel.dt.wrapper <- makeLearner("classif.rpart", minsplit = 3, cp = 0, xval = 0)
### my.wrapper <- makeLearner("classif.svm")
### my.wrapper <- makeLearner("classif.naiveBayes")

################################################
### compute performance with full set of features

sel.rdesc <- makeResampleDesc("CV", iters=5)

sel.repcv.full <- resample(sel.dt.wrapper, 
                       classif.task, 
                       sel.rdesc,
                       measures = sel.measure)

sel.full.mean <- mean(sel.repcv.full$measures.test[[2]])
cat('Repeated CV error % with full set of features =', 100 * round(sel.full.mean,3))

################################################



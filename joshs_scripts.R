library(readr)
library(dplyr)
library(mlbench)
library(tidyverse)
library(spFSR)
library(caret)
library(mlr)
library(kknn)

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

set.seed(12)

#partitioning training/validation : test (70:30)
train.index <- createDataPartition(ks$state, p = .7, list = FALSE)
train <- ks[ train.index,]
test  <- ks[-train.index,]


#template
task <- makeClassifTask(data = train, target = 'state', id = 'ks')

# Configure learners with probability type
lrns <- list(
  makeLearner("classif.naiveBayes", id = "NB", predict.type = 'prob'),
  makeLearner("classif.rpart", id = "rpart", predict.type = 'prob'),
  makeLearner("classif.randomForest", id = "randomForest", predict.type = 'prob'),
  makeLearner('classif.kknn', id= "knn", predict.type = 'prob')
)

# For naiveBayes, we can fine-tune Laplacian
ps.nb <- makeParamSet(
  makeNumericParam('laplace', lower = 0, upper = 30)
)

# For decision tree, fine-tune min.split i.e. minimum number of observations in a node for a split to be attempted
ps.rpart <- makeParamSet(
  makeIntegerParam('minsplit', lower = 1, upper = 50)
)


# For randomForest, fine-tune mtry i.e mumber of variables randomly 
# sampled as candidates at each split. Seeing as there are only 6 descriptive variables, smaller values will be considered
ps.rf <- makeParamSet(
  makeDiscreteParam('mtry', values = c(1,2,3))
)

# For kknn, we fine-tune k = 1 to 50 
ps.knn <- makeParamSet(
  makeIntegerParam('k', lower = 1, upper = 50)
)

# Configure tune control search to be random due to computational constraints and a 5-CV stratified sampling
ctrl  <- makeTuneControlRandom(maxit = 10)
rdesc <- makeResampleDesc("CV", iters = 5, stratify = TRUE)

#construct wrappers with with tuning constraints
tunedlrn.nb <- makeTuneWrapper(lrns[[1]], rdesc, mmce, ps.nb, ctrl)
tunedlrn.rpart <- makeTuneWrapper(lrns[[2]], rdesc, mmce, ps.rpart, ctrl)
tunedlrn.rf <- makeTuneWrapper(lrns[[3]], rdesc, mmce, ps.rf, ctrl)
tunedlrn.knn <- makeTuneWrapper(lrns[[4]], rdesc, mmce, ps.knn, ctrl)

#train models with tuned learners (this will take forever)
tunedMod.nb  <- train(tunedlrn.nb, task)
tunedMod.rpart  <- train(tunedlrn.rpart, task)
tunedMod.rf <- train(tunedlrn.rf, task)
tunedMod.knn  <- train(tunedlrn.knn, task)





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



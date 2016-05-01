require(plyr)  # load before dplyr
require(dplyr)
require(magrittr)
require(reshape2)
require(Metrics)
require(pROC)  # for caret twoClassSummary
require(caret)

#--------------------------------------
## NOTES
#.. BNP Paribas Cardif Claims Management
save.image("model.RData")
load("model.RData")
#load("explore.RData")
rm(list=ls()); gc()

#--------------------------------------
# SET GLOBAL OPTIONS
setwd("C:/Users/janet/Documents/GitLab/kaggle-bnp/src")
options(stringsAsFactors = FALSE)
options(dplyr.width = Inf)
set.seed(123)

#--------------------------------------
# HELPER FUNCTIONS

##====================================##
## MODELS: GBM, EXTRATREES, SVM, XGBOOST, NN, LOGISTIC

# create id to split full training set 
set.seed(123)
id.train <- createDataPartition(alldata[1:n.train,]$target, p=0.7, list=FALSE, times=1)

#--------------------------------------
# caret gbm
require(gbm)

# create training and testing set
train_full <- alldata.pp.lm[1:n.train,]
train_full$target <- as.factor(train_full$target)
levels(train_full$target) <- c("no","yes")   # 0=no 1=yes, levels need to be valid R col names
training <- tbl_df(train_full[id.train,-1])  # drop ID col
testing <- tbl_df(train_full[-id.train,-1])  # drop ID col

# set control for caret, repeated k-fold validation
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs=TRUE, summaryFunction=twoClassSummary)

set.seed(123)  # needs to be set right before train for resampling
mdl.gbm1 <- train(target ~., data = training,
                  method = "gbm",
                  metric = "ROC",
                  trControl = ctrl)
mdl.gbm1  # n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10
plot(mdl.gbm1)  # need a deeper tree

# tuning grid for gbm
grid.gbm <- expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

# run train with 4 CPUs -- ran for 8 hours did not finish
# require(doParallel); cl <- makeCluster(4); registerDoParallel(cl)  # register parallel backend
# set.seed(123)
# gbmFit2 <- train(target ~., data = training,
#                  method = "gbm",
#                  trControl = ctrl,
#                  metric = "ROC",
#                  tuneGrid = grid.gbm)
# stopCluster(cl); registerDoSEQ();  # un-register the parallel backend
# mdl.gbm2
# plot(mdl.gbm2)

# check logloss
predtest.gbm1 <- predict(mdl.gbm1, newdata=testing, type="prob")
logLoss(as.numeric(testing$target)-1, predtest.gbm1$yes)  # 0.476

#--------------------------------------
# neural net
require(neuralnet)

# create training and testing set
train_full <- alldata.pp.lm[1:n.train,]
preProcValues <- preProcess(train_full[-c(1,2)], method=c("center","scale"))
training <- predict(preProcValues, train_full[id.train,-1])
testing <- predict(preProcValues, train_full[-id.train,-1])

# usually one hidden layer is enough
# number of neurons should be between input layer and output layer size, usually 2/3 of input
form <- as.formula(paste("target~", paste(names(training)[2:137], collapse="+") ))
ptm <- proc.time()
set.seed(123)
nn <- neuralnet(form, data=training, hidden=1)
proc.time() - ptm

#--------------------------------------
# svm


#--------------------------------------
# extra trees

# drop columns from https://www.kaggle.com/mujtabaasif/bnp-paribas-cardif-claims-management/extratrees
# train %<>% select(-one_of(c('v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73',
#                             'v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109',
#                             'v110','v116','v117','v118','v119','v123','v124','v128')))

# n_estimators=850,max_features= 60,criterion= 'entropy',min_samples_split= 4,
# max_depth= 40, min_samples_leaf= 2, n_jobs = -1

#--------------------------------------
# xgboost
#.. run on processed logistic sublayer

# create matrices for xgboost training
xgtrain <- as.matrix(train[-1])  # drop target
xgtest <- as.matrix(test[-1])    # drop target
xgtrain_full <- as.matrix(train_full %>% dplyr::select(-ID, -target))
xgvalidation <- as.matrix(alldata.pp.lm[(n.train+1):nrow(alldata.pp.lm),] %>% dplyr::select(-ID, -target))

# xgboost unprocessed model
param1 <- list("objective" = "binary:logistic",  # binary classification 
               "eval_metric" = "logloss",        # match kaggle evaluation 
               "nthread" = 8,                    # number of threads to use
               "max_depth" = 10,                 # max depth of tree 
               "eta" = 0.05,                     # step size shrinkage (learning speed)
               #"gamma" = 0,                     # min loss reduction 
               "subsample" = 0.8,                # part of data instances to grow tree 
               "colsample_bytree" = 0.8,         # subsample ratio of columns when constructing each tree 
               "min_child_weight" = 1            # min sum of instance weight needed in a child 
)

require(xgboost)
bst.cv <- xgb.cv(data=xgtrain, param=param1, label=train$target,
                 nfold=5, nrounds=300, prediction=T)
(id.xg.logloss <- which.min(bst.cv$dt[, test.logloss.mean]))  # 122

# check with split test data
mdl.xg2 <- xgboost(param=param1, data=xgtrain, label=train$target,
                   nrounds=id.xg.logloss, verbose=T)
pred <- predict(mdl.xg2, xgtest)
head(test$target)
head(pred)
logLoss(test$target, pred)  # 0.4670923
# mdl.xgcv = xgb.dump(mdl.xg0, with.stats=TRUE)
# importance_matrix <- xgb.importance(dimnames(xgtrain)[[2]], model=mdl.xg0)
# gp <- xgb.plot.importance(importance_matrix)
# print(gp) 

# fit model with full training data
mdl.xg2 <- xgboost(param=param1, data=xgtrain_full, label=train_full$target,
                   nrounds=id.xg.logloss, verbose=T)

# prediction kaggle's final test data
pred <- predict(mdl.xg2, xgvalidation)
head(pred)
submission <- data.frame(ID=validation[,1], PredictedProb=pred)
write.csv(submission, "./submission/submission_xgpp1.csv", row.names=FALSE)


#--------------------------------------
# logistic
#.. "manual" model

train %<>% select(-one_of(c('v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73',
                            'v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109',
                            'v110','v116','v117','v118','v119','v123','v124','v128')))


#--------------------------------------
#.. weight together submissions (ensemble)
submission_xg002 <- read.csv("./submission/submission_xg002.csv")
pred2 <- rowMeans(cbind(pred, submission_xg002$PredictedProb))
submission <- data.frame(ID=validation[,1], PredictedProb=pred2)
write.csv(submission, "./submission/submission_xgpp1avg.csv", row.names=FALSE)


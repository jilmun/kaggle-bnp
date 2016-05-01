require(dplyr)
require(magrittr)
require(reshape2)
require(Metrics)
require(caret)
require(xgboost)
require(Matrix)
require(DiagrammeR)

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
balance_target <- function(dat, id) {
  count.target0 <- nrow(dat[id,] %>% filter(target==0))
  count.target1 <- nrow(dat[id,] %>% filter(target==1))
  dat$rowID <- 1:nrow(dat)
  id.target0 <- filter(dat[id,], target==0) %>% select(rowID) %>% unlist %>% as.numeric
  set.seed(123)
  id.bal <- sample(id.target0, count.target1-count.target0, replace=TRUE)
  id.all <- c(id, id.bal)
  message(table(dat[id.all,]$target))  # check equal rows for target=0,1
  dat$rowID <- NULL
  return(id.all)
}

##=====================================##
## MODEL: XGBOOST
## run on processed logistic sublayer
##=====================================##

# create id to split full training set 
set.seed(123)
id.train <- createDataPartition(alldata[1:n.train,]$target, p=0.7, list=FALSE, times=1)

# add extra samples to create balanced target in training data
# count.target0 <- nrow(alldata[id.train,] %>% filter(target==0))
# count.target1 <- nrow(alldata[id.train,] %>% filter(target==1))
# alldata$rowID <- 1:nrow(alldata)
# id.target0 <- filter(alldata[id.train,], target==0) %>% select(rowID) %>% unlist %>% as.numeric
# set.seed(123)
# id.bal <- sample(id.target0, count.target1-count.target0, replace=TRUE)
# id.train.bal <- c(id.train, id.bal)
# table(alldata.pp.lm[id.train.bal,]$target)  # check equal rows for target=0,1
# alldata$rowID <- NULL
# rm(count.target0)
# rm(count.target1)
# rm(id.target0)
id.train.bal <- balance_target(alldata, id.train)

# create training and testing set
train_full <- alldata.pp.lm[1:n.train,] %>% dplyr::select(-ID, -target)
train_full %<>% select(-one_of(c('v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73',
                                 'v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109',
                                 'v110','v116','v117','v118','v119','v123','v124','v128')))

# create matrices for xgboost training
xgtrain_full <- as.matrix(train_full)
xgtrain <- xgtrain_full[id.train.bal,]
xgtest <- xgtrain_full[-id.train,]
xgvalidation <- as.matrix(alldata.pp.lm[(n.train+1):nrow(alldata.pp.lm),] %>% 
                            dplyr::select(-ID, -target) %>%
                            dplyr::select(-one_of(c('v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73',
                                                    'v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109',
                                                    'v110','v116','v117','v118','v119','v123','v124','v128'))) ) 


# grid search for parameter tuning ----------------------------------------

xg.grid <- expand.grid(objective = "binary:logistic",  # binary classification 
                       eval_metric = "logloss",        # match kaggle evaluation 
                       
                       # control for model complexity
                       max_depth = c(4, 6, 8, 10),     # max depth of tree 
                       min_child_weight = 1,           # min sum of instance weight needed in a child 
                       gamma = 0,                      # min loss reduction required for split
                       
                       max_delta_step = 0,             # max delta step allowed for each tree's weight estimation
                       nround = 100,                   # fixed value 100-1000
                       eta = 0.05,                     # fixed value [2-10]/ntrees
                       
                       # control for robustness to noise
                       subsample = c(0.5, 0.75, 1), 
                       colsample_bytree = c(0.4, 0.6, 0.8, 1),
                       
                       score = 0,  # column to hold scores
                       stringsAsFactors = F)

# build an xgb.DMatrix object to speed up grid search
dtrain <- xgb.DMatrix(data = xgtrain, label = alldata.pp.lm[id.train.bal,]$target)

for (i in 1:nrow(xg.grid)) {
  message(paste0("\nRunning ",i," out of ",nrow(xg.grid),":\n"))
  mdl <- xgb.cv(data = dtrain,
                nfold = 5, 
                early.stop.round = 10,  # stop if the performance is worse consecutively for k rounds
                nthread = 8,            # number of cpu threads to use
                
                objective         = xg.grid[i,]$objective,
                eval_metric       = xg.grid[i,]$eval_metric,
                max_depth         = xg.grid[i,]$max_depth,
                min_child_weight  = xg.grid[i,]$max_depth,
                gamma             = xg.grid[i,]$gamma,        
                max_delta_step    = xg.grid[i,]$max_delta_step,    
                nround            = xg.grid[i,]$nround,          
                eta               = xg.grid[i,]$eta,            
                subsample         = xg.grid[i,]$subsample, 
                colsample_bytree  = xg.grid[i,]$colsample_bytree,
                
                prediction = T)
  
  # save logloss of the best iteration
  xg.grid[i,]$score <- min(mdl$dt[, test.logloss.mean])
}
xg.grid[which.min(xg.grid$score), ]
#          objective eval_metric max_depth min_child_weight gamma max_delta_step nround  eta subsample colsample_bytree    score
# 24 binary:logistic     logloss        10                1     0              0    100 0.05         1              0.6 0.498514

# tune `nround` and `eta` -------------------------------------------------

# set param to best grid search parameters
param.pp.lm.bal <- list("objective" = "binary:logistic",  # binary classification 
                        "eval_metric" = "logloss",        # match kaggle evaluation 
                        "max_depth" = 10,                 # max depth of tree 
                        "min_child_weight" = 1,           # min sum of instance weight needed in a child 
                        "gamma" = 0,                      # min loss reduction required for split
                        "max_delta_step" = 0,             # max delta step allowed for each tree's weight estimation
                        "subsample" = 1,                  # part of data instances to grow tree 
                        "colsample_bytree" = 0.6          # subsample ratio of columns when constructing each tree 
)

# run xgboost cv to find best nround
bst.cv <- xgb.cv(data = dtrain, param = param.pp.lm.bal,
                 nfold = 5, 
                 early.stop.round = 5,   # stop if the performance is worse consecutively for k rounds
                 nthread = 8,            # number of cpu threads to use
                 eta = 0.01,             # step size shrinkage (learning speed)
                 nround = 5000, 
                 prediction = T)
(id.bstll <- which.min(bst.cv$dt[, test.logloss.mean]))  
# nround=1000, eta=0.01, ll=0.440526+0.004006

# model with best parameters
bst <- xgboost(data = dtrain, param = param.pp.lm.bal,
               nthread = 4,      
               eta = 0.03,
               nround = id.bstll, 
               prediction = T)

# feature importance ------------------------------------------------------

# if overfit, reduce eta, increase nrounds at same time
xgb.importance(dimnames(xgtrain)[[2]], model=bst)
model <- xgb.dump(bst, with.stats = T)
xgb.plot.tree(feature_names = dimnames(xgtrain)[[2]], model = bst, n_first_tree = 2)


# check test hold out -----------------------------------------------------

pred <- predict(bst, xgtest)
head(test$target)
head(pred)
logLoss(test$target, pred)  # 0.4670923
# mdl.xgcv = xgb.dump(bst, with.stats=TRUE)
# importance_matrix <- xgb.importance(dimnames(xgtrain)[[2]], model=bst)
# gp <- xgb.plot.importance(importance_matrix)
# print(gp) 


# fit full train data -----------------------------------------------------

# add extra samples to create balanced target in training data
id.full.bal <- balance_target(alldata, 1:n.train)

xgtrain_full <- as.matrix(train_full)
xgtrain_full <- xgtrain_full[id.full.bal,]

bst.full <- xgboost(param=param.pp.lm.bal, data=xgtrain_full, label=alldata.pp.lm[id.full.bal,]$target,
                    nthread = 8,
                    early.stop.round = 5,  # train error shouldn't get worse
                    eta = 0.03,
                    nround = 5000, 
                    prediction = T,
                    verbose = T)

# final csv output --------------------------------------------------------

pred <- predict(bst.full, xgvalidation)
head(pred)
submission <- data.frame(ID=alldata[(n.train+1):nrow(alldata),1], PredictedProb=pred)
write.csv(submission, "../docs/subm_xgb_pp_lm_bal_all.csv", row.names=FALSE)

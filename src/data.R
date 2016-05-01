require(dplyr)
require(magrittr)
require(reshape2)
require(Metrics)
require(caret)

#--------------------------------------
## NOTES
#.. BNP Paribas Cardif Claims Management
save.image("explore.RData")
load("explore.RData")
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
## READ IN RAW DATA FILES
## train, test
train_orig <- read.csv("./data/train.csv")
validation <- read.csv("./data/test.csv")
n.train <- nrow(train_orig)
n.validation <- nrow(validation)

#--------------------------------------
# check for constant features
check_const <- sapply(train_orig, function(x) length(unique(x)))
check_const[check_const == 1]  # no features are constant

# check for near zero variance features -- use caret package
check_nzv <- nearZeroVar(train_orig %>% dplyr::select(-ID, -target), saveMetrics=T)
check_nzv[check_nzv$nzv == TRUE,]  # v3, v38, v74 have near 0 variance

#--------------------------------------
# check for best imputing method

alldata <- bind_rows(train_orig, validation)
alldata.pp <- alldata[,c("ID","target")]
global_impute_method <- data.frame(col=paste0("v",1:131), method=NA)
for (i in 1:131) {
  message(paste0("v",i,"... "))
  newcol <- dat_impute(unlist(alldata[,i+2]), train_orig$target, 1:n.train)
  alldata.pp <- cbind(alldata.pp, newcol[[1]])
  names(alldata.pp)[i+2] <- paste0("v",i)
  global_impute_method[i,2] <- newcol[[2]]
}
table(global_impute_method$method)

# test numeric v1 and categorical v30, v22 
# y <- train_orig$target
# var <- alldata$v22
# id.train <- 1:n.train

dat_impute <- function(var, y, id.train) {  # test v30 and v22 for categorical
  set.seed(123)
  
  if (is.numeric(var)) {
    impute.method <- c("min","max","med","0")
    var1 <- ifelse(is.na(var), min(var, na.rm=T), var)
    var2 <- ifelse(is.na(var), max(var, na.rm=T), var)
    var3 <- ifelse(is.na(var), median(var, na.rm=T), var)
    var4 <- ifelse(is.na(var), 0, var)

  } else {
    impute.method <- c("mode","random_sample","random_unif","-1")
    
    # check if validation categories are in training, was problem for v71
    cat.training <- table(var[id.train])
    cat.validation <- table(var[-id.train])
    cat.rare <- names(cat.validation)[!names(cat.validation) %in% names(cat.training)]
    
    # group rare categories, threshold is 100 count in training
    cat.rare <- unique(c(cat.rare, names(cat.training[cat.training<100])))
    
    # treat as "rare" group if total in training > 100, else treat as missing
    if (sum(cat.training[names(cat.training) %in% cat.rare]) > 100)
      var <- ifelse(var %in% cat.rare, "rare", var)
    else  
      var <- ifelse(var %in% cat.rare, "", var)  # treat as missing
    
    var.freq <- table(var)
    var.freq <- var.freq[rownames(var.freq) != ""]
    
    # only use 100 most frequent categories
    if (length(var.freq) > 100) {
      var.topN <- names(sort(var.freq, decreasing=T))[1:100]
      var <- ifelse(var %in% var.topN | var=="" | is.na(var), var, "other")
      var.freq <- table(var)
      var.freq <- var.freq[rownames(var.freq) != ""]
    }
    
    # impute with mode, most frequent category
    var.mode <- names(var.freq[var.freq==max(var.freq)])
    var1 <- ifelse(is.na(var) | var=="", var.mode, var)
    
    # impute with random category using actual distribution
    var.filled <- var[!is.na(var) & var!=""]
    var.random <- var.filled[sample(1:length(var.filled), 1)]
    var2 <- ifelse(is.na(var) | var=="", var.random, var)

    # impute with random category if actual distr is not representative
    var.categories <- rownames(var.freq)
    var.random <- var.categories[sample(1:length(var.categories), 1)]
    var3 <- ifelse(is.na(var) | var=="", var.random, var)
    
    # impute with its own category
    var4 <- ifelse(is.na(var) | var=="", "-1", var)
  }
  mfit <- summary(lm(y~var1[id.train]))$sigma
  mfit <- c(mfit, summary(lm(y~var2[id.train]))$sigma)
  mfit <- c(mfit, summary(lm(y~var3[id.train]))$sigma)
  mfit <- c(mfit, summary(lm(y~var4[id.train]))$sigma)
  myvars <- data.frame(var1, var2, var3, var4)
  id.bst <- which.min(mfit)
  var.bst <- myvars[, id.bst]
  message(impute.method[id.bst])  # output method
  return(list(Var=var.bst, Method=impute.method[id.bst]))
}


#--------------------------------------
# standardize features with lm prob results
#.. returns warnings:
#..  glm.fit: fitted probabilities numerically 0 or 1 occurred (linearly separable?) -- v10
#..  algorithm did not converge

options(warn = 0)  # warn=2 changes warnings to errors to break loop, warn=0 (default)
alldata.pp.lm <- alldata[,c("ID","target")]
global_warning_lm <- data.frame(col=paste0("v",1:131), warn=FALSE, msg="")
for (i in 1:131) {
  tt <- myTryCatch(sublayer_lm(alldata.pp[,c(2,i+2)], 1:n.train))  # custom warning catch
  newcol <- tt$value
  if(!is.null(tt$warning)) {
    global_warning_lm[i,2] <- TRUE
    global_warning_lm[i,3] <- tt$warning$message
  }
  alldata.pp.lm <- cbind(alldata.pp.lm, newcol)
  names(alldata.pp.lm)[i+2] <- paste0("v",i)
}
global_warning_lm %>% filter(warn == TRUE)
check_na <- alldata.pp.lm[rowSums(is.na(alldata.pp.lm %>% select(-target)))>0,]

# test a specific column
# dat <- alldata.pp[,c("target","v10")] 

sublayer_lm <- function(dat, id.train) {
  require(glmnet)
  message(paste0("Processing ", names(dat)[2]))
  dat$target <- as.factor(dat$target)
  names(dat)[2] <- "x"
  if (is.numeric(dat$x)) {
    mdl.lm <- glm(target~poly(x,5), data=dat[id.train,], family=binomial)
  } else {
    dat$x <- as.factor(dat$x)
    mdl.lm <- glm(target~., data=dat[id.train,], family=binomial)
  }
  mdl.prob <- predict(mdl.lm, newdata=dat, type="response")
  
  # rescale by subtracting min and divide by range
  prob.min <- range(mdl.prob)[1] 
  prob.max <- range(mdl.prob)[2]
  mdl.prob.adj <- (mdl.prob - prob.min)  # /(prob.max-prob.min) # if small range, then less weight?
  return(mdl.prob.adj)
}

# custom tryCatch to return result and warnings
myTryCatch <- function(expr) {
  warn <- err <- NULL
  value <- withCallingHandlers(
    tryCatch(expr, error=function(e) {
      err <<- e
      NULL
    }), warning=function(w) {
      warn <<- w
      invokeRestart("muffleWarning")
    })
  list(value=value, warning=warn, error=err)
}


#--------------------------------------
# create aggregate variables 

# average of lm probabilities
alldata.pp.lm$feat_avg <-apply(alldata.pp.lm %>% dplyr::select(-ID, -target), 1, mean)

# std dev of lm probabilities
alldata.pp.lm$feat_sd <- apply(alldata.pp.lm %>% dplyr::select(-ID, -target), 1, sd) 

# count of missing features/total
alldata.tmp <- alldata
alldata.tmp[alldata.tmp == ""] <- NA  # replace blanks with NA
alldata.pp.lm$feat_missingTot <- apply(is.na(alldata.tmp %>% dplyr::select(-ID, -target)), 1, sum)
alldata.pp.lm$feat_missingNum <- apply(is.na(alldata %>% dplyr::select(-ID, -target)), 1, sum)
rm(alldata.tmp)

# create NA/blank patterns
alldata.pp.lm$feat_NA <- apply(alldata.tmp %>% dplyr::select(-ID, -target), 1, fn_NApattern)
fn_NApattern <- function(myrow) {
  return(paste(is.na(myrow)+0, collapse=""))
}
unique(alldata.pp.lm$feat_NA) %>% length()  # 252
check_tmp <- table(alldata.pp.lm$feat_NA)   
newcol <- dat_impute(alldata.pp.lm$feat_NA, train_orig$target, 1:n.train)
alldata.pp.lm$feat_NA.pp <- newcol[[1]]
alldata.pp.lm$feat_NA.pp <- sublayer_lm(alldata.pp.lm[,c("target","feat_NA.pp")] , 1:n.train)
global_NA_pattern <- alldata.pp.lm %>% group_by(feat_NA, feat_NA.pp) %>% tally()  # save encodings
alldata.pp.lm$feat_NA <- NULL  # delete original column with long strings

## return to work here -- more exploration? ##


#--------------------------------------
# run processed data through original xgboost cross validation script

#.. split training set 
train_full <- alldata.pp.lm[1:n.train,]
set.seed(123)
id <- createDataPartition(alltrain$target, p=0.7, list=FALSE, times=1)
train <- tbl_df(train_full[id,-1])  # drop ID col
test <- tbl_df(train_full[-id,-1])  # drop ID col

#.. create matrices for xgboost training
xgtrain <- as.matrix(train[-1])  # drop target
xgtest <- as.matrix(test[-1])    # drop target
xgtrain_full <- as.matrix(train_full %>% dplyr::select(-ID, -target))
xgvalidation <- as.matrix(alldata.pp.lm[(n.train+1):nrow(alldata.pp.lm),] %>% dplyr::select(-ID, -target))

#.. xgboost unprocessed model
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

#.. check with split test data
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

#.. fit model with full training data
mdl.xg2 <- xgboost(param=param1, data=xgtrain_full, label=train_full$target,
                   nrounds=id.xg.logloss, verbose=T)

#.. prediction kaggle's final test data
pred <- predict(mdl.xg2, xgvalidation)
head(pred)
submission <- data.frame(ID=validation[,1], PredictedProb=pred)
write.csv(submission, "./submission/submission_xgpp1.csv", row.names=FALSE)

#.. avg submission with unprocessed xgboost
submission_xg002 <- read.csv("./submission/submission_xg002.csv")
pred2 <- rowMeans(cbind(pred, submission_xg002$PredictedProb))
submission <- data.frame(ID=validation[,1], PredictedProb=pred2)
write.csv(submission, "./submission/submission_xgpp1avg.csv", row.names=FALSE)
#--------------------------------------
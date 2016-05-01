# bnp paribas cardis

# dependencies =================================================================#
# source("https://git.io/vz9AS")

pacman::p_load(caret, Matrix, xgboost)
pacman::p_load(plyr, dplyr, tidyr, stringr, readr, purrr)

setwd("C:/Users/tyoko/Dropbox/bnp_paribas_cardif")
# setwd("E:/lol/bnp_paribas_cardis")

# data =========================================================================#
tmp <- tempdir()
dir.files <- list.files("data")

for (i in dir.files) {
  unzip(paste("data", i, sep = "/"), exdir = tmp)
}

dir.tmp <- list.files(tmp, pattern = ".csv", full.names = T)
list.tmp <- lapply(dir.tmp, read_csv)

names(list.tmp) <- str_replace_all(list.files(tmp, pattern = ".csv"), ".csv", "")

invisible(lapply(names(list.tmp), function(x)
  assign(x, list.tmp[[x]], envir = .GlobalEnv)))

rm(tmp, dir.files, dir.tmp, list.tmp)
gc()

# preprocess ===================================================================#
target <- train$target
test$target <- -1

dat_impute <- function(var, y, id.train) {
  
  if (is.numeric(var)) {
    
    # note: round to remove randomly added noise
    var <- round(var, digits = 5)
    
  } else {
    cat.training <- table(var[id.train])
    cat.validation <- table(var[-id.train])
    cat.rare <- names(cat.validation)[!names(cat.validation) %in% names(cat.training)]
    
    # note: group rare categories, threshold is 100 count in training
    cat.rare <- unique(c(cat.rare, names(cat.training[cat.training<100])))
    
    # note: treat as "rare" group if total in training > 100, else treat as missing
    if (sum(cat.training[names(cat.training) %in% cat.rare]) > 100)
      var <- ifelse(var %in% cat.rare, "rare", var)
    else
      var <- ifelse(var %in% cat.rare, "missing", var)  # treat as missing
    
    var.freq <- table(var)
    var.freq <- var.freq[rownames(var.freq) != "missing"]
    
    # note: only use 100 most frequent categories
    if (length(var.freq) > 100) {
      var.topN <- names(sort(var.freq, decreasing=T))[1:100]
      var <- ifelse(var %in% var.topN | var=="" | is.na(var), var, "other")
    }
  }
  myvars <- data.frame(var, stringsAsFactors = F)
  
  return(list(Var=myvars))
}

n.train <- nrow(train)

alldata <- bind_rows(train, test)
alldata.pp <- alldata[,c("ID","target")]
var.list <- names(alldata[3:133])

for (i in 1:131) {
  message(paste0("v",i,"... "))
  newcol <- dat_impute(unlist(alldata[,i+2]), train$target, 1:n.train)
  alldata.pp <- cbind(alldata.pp, newcol[[1]])
  names(alldata.pp)[i+2] <- paste0(var.list[i])
}

train <- alldata.pp[alldata.pp$ID %in% train$ID, ]
test <- alldata.pp[alldata.pp$ID %in% test$ID, ]

train2 <- alldata.pp[alldata.pp$ID %in% train$ID, ]
test2 <- alldata.pp[alldata.pp$ID %in% test$ID, ]

# note: leave-one-fold-out categorical encoding [feifeiyu1204@gmail]
set.seed(96707)
folds <- createMultiFolds(target, k = 10, times = 1)

gc()

ohe.list <- names(train[, sapply(train, is.character)])
ohe.vars <- c("ID", "target", ohe.list)

train <- train[, (names(train) %in% ohe.vars)]
test  <- test[, (names(test) %in% ohe.vars)]

train <- train %>% replace(is.na(.), "missing")
test <- test %>% replace(is.na(.), "missing")

lofoEncoder <- function(stat) {
  print(stat)
  for (j in ohe.list) {
    list.x <- list()
    # note: mean encoding
    if(stat=="_mean") {
      for (i in seq(1, length(folds))) {
        lofo.ids <- folds[[i]]
        
        x <- train[lofo.ids, ] %>%
          group_by_(j) %>%
          summarise(var_new = mean(target, na.rm = T))
        y <- train[-lofo.ids, ] %>% left_join(x) %>% select_("ID", j, "var_new")
        list.x[[i]] <- y
      }
    } else if (stat=="_n") {
      for (i in seq(1, length(folds))) {
        lofo.ids <- folds[[i]]
        
        x <- train[lofo.ids, ] %>%
          group_by_(j) %>%
          summarise(var_new = n())
        y <- train[-lofo.ids, ] %>% left_join(x) %>% select_("ID", j, "var_new")
        list.x[[i]] <- y
      }
    } else if (stat=="_sd") {
      for (i in seq(1, length(folds))) {
        lofo.ids <- folds[[i]]
        x <- train[lofo.ids, ] %>%
          group_by_(j) %>%
          summarise(var_new = sd(target, na.rm = T))
        y <- train[-lofo.ids, ] %>% left_join(x) %>% select_("ID", j, "var_new")
        list.x[[i]] <- y
      }
    } else {
      print("specify encoding statistic")
    }
    list.x <- do.call(rbind, list.x)
    
    list.y <- list.x %>%
      group_by_(j) %>%
      summarise(var_new = mean(var_new, na.rm = T))
    names(list.x)[3] <- paste0(j, stat)
    train <- train %>% left_join(list.x[, c(1,3)], by = "ID")
    names(list.y)[2] <- paste0(j, stat)
    test <- test %>% left_join(list.y)
  }
  ldf <- list(train = train, test = test)
  return(ldf)
}

ldf_mean <- lofoEncoder("_mean")
train <- ldf_mean$train
test  <- ldf_mean$test
rm(ldf_mean)

ldf_n <- lofoEncoder("_n")
train <- ldf_n$train
test  <- ldf_n$test
rm(ldf_n)

ldf_sd <- lofoEncoder("_sd")
train <- ldf_sd$train
test  <- ldf_sd$test
rm(ldf_sd)

gc()
names(train)

train <- train[, !(names(train) %in% c("target"))]
names(train)

test <- test[, !(names(test) %in% c("target"))]
names(test)

train <- train[, !(names(train) %in% ohe.list)]
test  <- test[, !(names(test) %in% ohe.list)]

# concatenate train/test set ===================================================#
# pp_data <- read.csv("alldata_pp_lm.csv", stringsAsFactors = F)
# names(pp_data)

# test2 <- pp_data[pp_data$ID %in% test$ID, ]
# train2 <- pp_data[pp_data$ID %in% train$ID, ]

train2 <- train2[, !(names(train2) %in% ohe.list)]
test2  <- test2[, !(names(test2) %in% ohe.list)]

train <- train %>%
  map_if(is.character, as.factor) %>%
  map_if(is.factor, as.integer) %>% data.frame

test <- test %>%
  map_if(is.character, as.factor) %>%
  map_if(is.factor, as.integer) %>% data.frame

train_df <- left_join(train, train2)
test_df  <- left_join(test, test2)

names(train_df)

save(train_df, file="train_df.Rda")
save(test_df, file="test_df.Rda")
save(sample_submission, file="sample_submission.Rda")

load("train_df.Rda")
load("test_df.Rda")

train_id <- train_df$ID
test_id  <- test_df$ID

train_df <- train_df %>% subset(., select = -c(ID, target))
test_df  <- test_df %>% subset(., select = -c(ID, target))
names(test_df)

sum(is.na(train_df))
sum(is.na(test_df))
gc()

# level 1 stack ================================================================#
median_impute <- preProcess(train_df, method = c("medianImpute"))
train_df <- predict(median_impute, train_df)

test_df <- predict(median_impute, test_df)

stx.folds <- 10
stx_cv <- createFolds(target, k = stx.folds)

# stack 1: xgboost with AUC ----------------------------------------------------#

stx1_stack <- data.frame()
stx1_test  <- data.frame()

stx1_test_df  <- xgb.DMatrix(data.matrix(test_df))

for (i in seq(1, length(stx_cv)) ) {
  set.seed(123 * i)
  
  fold.ids <- setdiff(1:nrow(train_df), stx_cv[[i]])
  print(length(fold.ids))
  print(names(stx_cv)[i])
  
  # note: id
  stx1_id      <- train_id[fold.ids]
  stx1_fold_id <- train_id[-fold.ids]
  
  # note: create training and hold-out (fold)
  stx1_train_df <- xgb.DMatrix(data.matrix(train_df[fold.ids, ]), label = target[fold.ids])
  stx1_fold_df  <- xgb.DMatrix(data.matrix(train_df[-fold.ids, ]))
  
  # note: hypertuning
  param1 <- list("objective" = "binary:logistic",
                 "eval_metric" = "auc",
                 "eta" = 0.01, #0.01
                 "max_depth" = 8,
                 "subsample" = 0.95,
                 "colsample_bytree" = 0.45,
                 "lambda" = 0,
                 "alpha" = 0.2,
                 "min_child_weight" = 1,
                 "gamma" = 1)
  
  # note: run CV for first iteration
  if (i == 1) {
    cv.nround = 3000
    n_proc = 4
    bst.cv = xgb.cv(param = param1,
                    data = stx1_train_df, 
                    nfold = stx.folds,
                    nrounds = cv.nround,
                    early.stop.round = 10)
  }
  # note: traing consequential folds train xgboost
  stx1_mdl <- xgboost(data = stx1_train_df, 
                      param = param1,
                      nround = which.max(bst.cv$test.auc.mean))
  
  # note: predict for fold
  stx1_fold_var <- predict(stx1_mdl, stx1_fold_df)
  stx1_fold_mat <- matrix(stx1_fold_var, nrow=nrow(train_df[-fold.ids, ]), ncol=1, byrow=T)
  stx1_fold_pred <- as.data.frame(stx1_fold_mat)
  names(stx1_fold_pred) <- paste0("stack1_", "target")
  stx1_fold_pred <- mutate(stx1_fold_pred, id = unique(stx1_fold_id))
  stx1_fold_pred <- stx1_fold_pred[c("id", paste0("stack1_", "target"))] # swap
  stx1_stack <- bind_rows(stx1_stack, stx1_fold_pred)
  
  # note: test
  stx1_test_var <- predict(stx1_mdl, stx1_test_df)
  stx1_test_mat <- matrix(stx1_test_var, nrow=nrow(test_df), ncol=1, byrow=T)
  stx1_test_pred <- as.data.frame(stx1_test_mat)
  names(stx1_test_pred) <- paste0("stack1_", "target")
  stx1_test_pred <- mutate(stx1_test_pred, id = unique(test_id))
  stx1_test_pred <- stx1_test_pred[c("id", paste0("stack1_", "target"))] # swap
  stx1_test <- bind_rows(stx1_test, stx1_test_pred)
}
stx1_test <- stx1_test %>%
  group_by(id) %>%
  summarise_each(funs(mean))

save(stx1_stack, file="stx1_stack.Rda")
save(stx1_test, file="stx1_test.Rda")

# stack 2: lasso ---------------------------------------------------------------#
pacman::p_load(doParallel)
pacman::p_load(glmnet)

stx2_stack <- data.frame()
stx2_test  <- data.frame()

for (i in seq(1, length(stx_cv)) ) {
  set.seed(123 * i)
  
  fold.ids <- setdiff(1:nrow(train_df), stx_cv[[i]])
  print(length(fold.ids))
  print(names(stx_cv)[i])
  
  # note: id
  stx2_id      <- train_id[fold.ids]
  stx2_fold_id <- train_id[-fold.ids]
  
  # note: target
  stx2_target <- target[-fold.ids]

  stx2_fold_df  <- train_df[-fold.ids, ]
  
  # note: create training and hold-out (fold)
  registerDoParallel(4)
  # note: run CV for first iteration
  if (i == 1) {
    stx2_cv <- cv.glmnet(as.matrix(train_df[fold.ids, ]), as.factor(target[fold.ids]), 
                          family="binomial", type.measure="auc", nfolds = stx.folds,
                          alpha=1, parallel = T)  # alpha: 0=ridge, 1=lasso
    bst.lambda <- stx2_cv$lambda.min # best lambda with cv
  }
  
  stx2_mdl <- glmnet(as.matrix(train_df[fold.ids, ]),as.factor(target[fold.ids]),
                     family="binomial",lambda=bst.lambda)
  
  # note: predict for fold
  stx2_fold_var <- predict(stx2_mdl, s=bst.lambda, newx=as.matrix(stx2_fold_df), type="response")
  stx2_fold_mat <- matrix(stx2_fold_var, nrow=nrow(train_df[-fold.ids, ]), ncol=1, byrow=T)
  stx2_fold_pred <- as.data.frame(stx2_fold_mat)
  names(stx2_fold_pred) <- paste0("stack2_", "target")
  stx2_fold_pred <- mutate(stx2_fold_pred, id = unique(stx2_fold_id))
  stx2_fold_pred <- stx2_fold_pred[c("id", paste0("stack2_", "target"))] # swap
  stx2_stack <- bind_rows(stx2_stack, stx2_fold_pred)
  
  # note: test
  stx2_test_var <- predict(stx2_mdl, s=bst.lambda, newx=as.matrix(test_df), type="response")
  stx2_test_mat <- matrix(stx2_test_var, nrow=nrow(test_df), ncol=1, byrow=T)
  stx2_test_pred <- as.data.frame(stx2_test_mat)
  names(stx2_test_pred) <- paste0("stack2_", "target")
  stx2_test_pred <- mutate(stx2_test_pred, id = unique(test_id))
  stx2_test_pred <- stx2_test_pred[c("id", paste0("stack2_", "target"))] # swap
  stx2_test <- bind_rows(stx2_test, stx2_test_pred)
}
stx2_test <- stx2_test %>%
  group_by(id) %>%
  summarise_each(funs(mean))

save(stx2_stack, file="stx2_stack.Rda")
save(stx2_test, file="stx2_test.Rda")


# stack 3: KNN -----------------------------------------------------------------# 
pacman::p_load(ranger)

stx3_stack <- data.frame()
stx3_test  <- data.frame()

for (i in seq(1, length(stx_cv)) ) {
  set.seed(123 * i)
  
  fold.ids <- setdiff(1:nrow(train_df), stx_cv[[i]])
  print(length(fold.ids))
  print(names(stx_cv)[i])
  
  # note: id
  stx3_id      <- train_id[fold.ids]
  stx3_fold_id <- train_id[-fold.ids]
  
  # note: target
  stx3_target <- target[-fold.ids]
  
  stx3_d <- train_df[fold.ids, ]
  stx3_d$target <- target[fold.ids]
  
  stx3_fold_df  <- train_df[-fold.ids, ]
  
  # note: create training and hold-out (fold)
  # note: run CV for first iteration
  # if (i == 1) {

  # }
  stx3_mdl <- ranger(target ~ ., data = stx3_d, write.forest = TRUE)
  
  # note: predict for fold
  stx3_fold_var <- predict(stx3_mdl, dat=stx3_fold_df)
  stx3_fold_mat <- matrix(stx3_fold_var, nrow=nrow(train_df[-fold.ids, ]), ncol=1, byrow=T)
  stx3_fold_pred <- as.data.frame(stx3_fold_mat)
  names(stx3_fold_pred) <- paste0("stack3_", "target")
  stx3_fold_pred <- mutate(stx3_fold_pred, id = unique(stx3_fold_id))
  stx3_fold_pred <- stx3_fold_pred[c("id", paste0("stack3_", "target"))] # swap
  stx3_stack <- bind_rows(stx3_stack, stx3_fold_pred)
  
  # note: test
  stx3_test_var <- predict(stx3_mdl, dat=test_df)
  stx3_test_mat <- matrix(stx3_test_var, nrow=nrow(test_df), ncol=1, byrow=T)
  stx3_test_pred <- as.data.frame(stx3_test_mat)
  names(stx3_test_pred) <- paste0("stack3_", "target")
  stx3_test_pred <- mutate(stx3_test_pred, id = unique(test_id))
  stx3_test_pred <- stx3_test_pred[c("id", paste0("stack3_", "target"))] # swap
  stx3_test <- bind_rows(stx3_test, stx3_test_pred)
}
stx3_test <- stx3_test %>%
  group_by(id) %>%
  summarise_each(funs(mean))

save(stx3_stack, file="stx3_stack.Rda")
save(stx3_test, file="stx3_test.Rda")

# model data ===================================================================#
load("train_df.Rda")
load("test_df.Rda")
load("sample_submission.Rda")

load("stx1_stack.Rda")
load("stx1_test.Rda")
load("stx2_stack.Rda")
load("stx2_test.Rda")
load("stx3_stack.Rda")
load("stx3_test.Rda")

head(stx1_stack)
train_df <- train_df %>% left_join(stx1_stack, by=c("ID"="id"))
train_df <- train_df %>% left_join(stx2_stack, by=c("ID"="id"))

test_df <- test_df %>% left_join(stx1_test, by=c("ID"="id"))
test_df <- test_df %>% left_join(stx2_test, by=c("ID"="id"))

target <- train_df$target

train_df <- train_df %>% replace(is.na(.), -99999)
test_df <- test_df %>% replace(is.na(.), -99999)
names(test_df)

train_id <- train_df$ID
test_id  <- test_df$ID

train_df <- train_df %>% subset(., select = -c(ID, target))
test_df  <- test_df %>% subset(., select = -c(ID, target))
names(test_df)

sum(is.na(train_df))
sum(is.na(test_df))

# note: evaluation function
LogLoss.score <- function(actual, predicted) {
  ll <- -1 / length(actual) * (sum((actual * log(predicted) + (1 - actual) * log(1 - predicted))))
  return(ll)
}

LogLoss.eval <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  retval <- LogLoss.score(labels, preds)
  return (list(metric = "LogLoss", value = retval))
}

# grid search for parameter tuning
xgtrain <- xgb.DMatrix(data = data.matrix(train_df), label = target)
# rm(train_df2)
# rm(test_df2)
tuneXGB <- function(df, tune_grid) {
  my_list <- list()
  for (k in 1:nrow(tune_grid)) {
    message(paste0("\nRunning ",k," out of ",nrow(tune_grid),":\n"))
    prm <- tune_grid[k,]
    print(prm)
    print(system.time({
      n_proc <- 4
      set.seed(2016)
      md <- xgb.cv(data = df,
                   nthread = n_proc,
                   objective = "binary:logistic",
                   nround = prm$nround, #500
                   eta = prm$eta,
                   colsample_bytree = prm$colsample_bytree,
                   subsample = prm$subsample,
                   max_depth = prm$max_depth,
                   gamma = prm$gamma,
                   eval_metric = "logloss",
                   min_child_weight = prm$min_child_weight,
                   alpha=prm$alpha,
                   early_stop_round = 25,
                   nfold = 5,
                   print.every.n = 25,
                   maximize = F)
      my_list[[k]] <- data.frame(prm, iter = which(md$test.logloss.mean == min(md$test.logloss.mean)), score = min(md$test.logloss.mean))
    }))
  }
  return(my_list)
}

tune_grid <- expand.grid(eta = c(0.1),
                         max_depth = c(6),
                         min_child_weight = c(1),
                         subsample =  c(1),
                         colsample_bytree = c(0.3, 0.4,0.5,0.6), #0.55
                         gamma = c(0),
                         alpha=c(0),
                         lambda = c(0),
                         nround=75) # note: quick iterations just to get parameters

mdl <- tuneXGB(xgtrain, tune_grid)
grid_final <- do.call(rbind, mdl)
arrange(grid_final, score, iter)

# note: results from CV
# 0.459838
# 0.459584

# note: bagging
iters <- 15

test_stack  <- data.frame()

test_DM <- xgb.DMatrix(data = data.matrix(test_df))

for (i in 1:iters ) {
  print(paste0("Fold", i))
  
  train_DM  <- xgb.DMatrix(data = data.matrix(train_df), label = target)
  
  param <- list("objective" = "binary:logistic",
                "eval_metric" = "logloss",
                "eta" = 0.01, 
                "max_depth" = 6,
                "subsample" = 1,
                "colsample_bytree" = 0.9,
                "lambda" = 0,
                "alpha" = 0.2,
                "min_child_weight" = 1,
                "gamma" = 1)
  
  set.seed(2016 * i)
  xgb <- xgb.train(params = param,
                   data = train_DM,
                   nrounds = 496, 
                   print.every.n = 25,
                   maximize = F)
  
  test_ <- predict(xgb, test_DM)
  test_pred_df <- data.frame(dfb_dac_lag_pred = test_)
  test_pred_df <- mutate(test_pred_df, id =  test_id)
  test_pred_df <- test_pred_df[c("id", "dfb_dac_lag_pred")]
  test_stack <- bind_rows(test_stack, test_pred_df)
}
str(train_stack)
test_stack <- test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

gc()

# note: feature importance
mdl.xgcv <- xgb.dump(xgb, with.stats=TRUE)
importance_matrix <- xgb.importance(dimnames(train_df)[[2]], model=xgb)
gp <- xgb.plot.importance(importance_matrix)
print(gp)
importance_matrix[1:20]
# write_csv(importance_matrix, "important_features.csv")

# note: create submission
submission <- NULL
id_mtx <-  matrix(test_id, 1)[rep(1,1), ]
ids <- c(id_mtx)
submission$id <- ids

submission <- data.frame(submission) %>% left_join(test_stack)
submission <- as.data.frame(submission)
head(submission)
names(submission) <- c("id", "PredictedProb")

write.csv(submission, "submission_with_cv2.csv", quote=FALSE, row.names = FALSE)
# http://hi.healthinspections.us/hawaii/API/index.cfm/inspectionsData/MzcwMg==

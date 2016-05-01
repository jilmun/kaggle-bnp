# bnp paribas cardis

#=====================================#
# dependencies
# source("https://git.io/vz9AS")

pacman::p_load(caret, Matrix, xgboost, FeatureHashing)
pacman::p_load(plyr, dplyr, tidyr, stringr, readr, purrr)

setwd("C:/Users/tyoko/Dropbox/bnp_paribas_cardif")
# setwd("E:/lol/bnp_paribas_cardis")

#=====================================#
# data
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

#=====================================#
# preprocess
target <- train$target
test$target <- -1

full <- rbind_list(train, test)

#-------------------------------------#
# note: handle missing data
train <- train %>%
  mutate(row_miss = rowSums(is.na(.)))

test <- test %>%
  mutate(row_miss = rowSums(is.na(.)))

#-------------------------------------#
# note: preprocess numeric variables
highCorrRemovals <- c("v8", "v23", "v25", "v36", "v37", "v46",
                      "v51", "v53", "v54", "v63", "v73", "v81",
                      "v82", "v89", "v92", "v95", "v105", "v107",
                      "v108", "v109", "v116", "v117", "v118",
                      "v119", "v123", "v124", "v128")

train <- train[, !(names(train) %in% highCorrRemovals)]
test  <- test[, !(names(test) %in% highCorrRemovals)]

# feature importance
train_fs <- train[, sapply(train, is.numeric)]
drops <- c("ID","target")

train_fs <- train_fs[, !(names(train_fs) %in% drops)]

pacman::p_load(mRMRe)

ind <- sapply(train_fs, is.integer)
train_fs[ind] <- lapply(train_fs[ind], as.numeric)

dd <- mRMR.data(data = train_fs %>% select(-v50))

feats <- mRMR.classic(data = dd, target_indices = c(ncol(train_fs)), feature_count = 10)

variableImportance <-data.frame('importance'=feats@mi_matrix[nrow(feats@mi_matrix),])
variableImportance$feature <- rownames(variableImportance)
row.names(variableImportance) <- NULL
variableImportance <- na.omit(variableImportance)
variableImportance <- variableImportance[order(variableImportance$importance, decreasing=TRUE),]
print(variableImportance)

variableImportance <- variableImportance %>%
  filter(importance > 0.010) %>%
  select(feature)

nrow(variableImportance)

train <- train %>%
  mutate(row_miss = rowSums(is.na(.)))

test <- test %>%
  mutate(row_miss = rowSums(is.na(.)))

# # zero counts
# feature.names <- names(train)
# N <- ncol(train)
# train$train_ZeroCount <- rowSums(train[,feature.names]== 0) / N
# train$train_Below0Count <- rowSums(train[,feature.names] < 0) / N
# 
# feature.names <- names(test)
# N <- ncol(test)
# test$test_ZeroCount <- rowSums(test[,feature.names]== 0) / N
# test$test_Below0Count <- rowSums(test[,feature.names] < 0) / N

# note: interaction variables
train_fs <- train[, sapply(train, is.numeric)]
test_fs <- test[, sapply(test, is.numeric)]

train_fs2 <- train_fs[, (names(train_fs) %in% variableImportance$feature)]
test_fs2 <- test_fs[, (names(test_fs) %in% variableImportance$feature)]

length(train_fs2)
length(test_fs2)

for (i in 1:23) {
  for (j in (i + 1) : 24) {
    var.x <- colnames(train_fs2)[i]
    var.y <- colnames(train_fs2)[j]
    var.new <- paste0(var.x, 'min', var.y)
    x <- train_fs2[, i] - train_fs2[, j]
    train_fs2 <- cbind(train_fs2, x)
    names(train_fs2)[length(train_fs2)] <- paste0(var.new)  }
}

for (i in 1:23) {
  for (j in (i + 1) : 24) {
    var.x <- colnames(test_fs2)[i]
    var.y <- colnames(test_fs2)[j]
    var.new <- paste0(var.x, 'min', var.y)
    x <- test_fs2[, i] - test_fs2[, j]
    test_fs2 <- cbind(test_fs2, x)
    names(test_fs2)[length(test_fs2)] <- paste0(var.new)
  }
}

for (i in 1:23) {
  for (j in (i + 1) : 24) {
    var.x <- colnames(train_fs2)[i]
    var.y <- colnames(train_fs2)[j]
    var.new <- paste0(var.x, 'add', var.y)
    x <- train_fs2[, i] + train_fs2[, j]
    train_fs2 <- cbind(train_fs2, x)
    names(train_fs2)[length(train_fs2)] <- paste0(var.new)  }
}

for (i in 1:23) {
  for (j in (i + 1) : 24) {
    var.x <- colnames(test_fs2)[i]
    var.y <- colnames(test_fs2)[j]
    var.new <- paste0(var.x, 'add', var.y)
    x <- test_fs2[, i] + test_fs2[, j]
    test_fs2 <- cbind(test_fs2, x)
    names(test_fs2)[length(test_fs2)] <- paste0(var.new)
  }
}

for (i in 1:23) {
  for (j in (i + 1) : 24) {
    var.x <- colnames(train_fs2)[i]
    var.y <- colnames(train_fs2)[j]
    var.new <- paste0(var.x, 'product', var.y)
    x <- train_fs2[, i] * train_fs2[, j]
    train_fs2 <- cbind(train_fs2, x)
    names(train_fs2)[length(train_fs2)] <- paste0(var.new)  }
}

for (i in 1:23) {
  for (j in (i + 1) : 24) {
    var.x <- colnames(test_fs2)[i]
    var.y <- colnames(test_fs2)[j]
    var.new <- paste0(var.x, 'product', var.y)
    x <- test_fs2[, i] * test_fs2[, j]
    test_fs2 <- cbind(test_fs2, x)
    names(test_fs2)[length(test_fs2)] <- paste0(var.new)
  }
}

train_fs2 <- train_fs2[, !(names(train_fs2) %in% variableImportance$feature)]
train_fs <- cbind(train_fs, train_fs2)

test_fs2 <- test_fs2[, !(names(test_fs2) %in% variableImportance$feature)]
test_fs <- cbind(test_fs, test_fs2)

train_fs <- train_fs %>% replace(is.na(.), -99999)
test_fs <- test_fs %>% replace(is.na(.), -99999)

# train_num <- full.pp2[full.pp2$ID %in% train$ID, ]
# test_num  <- full.pp2[full.pp2$ID %in% test$ID, ]

train_fs <- train_fs[, !(names(train_fs) %in% drops)]
test_fs  <- test_fs[, !(names(test_fs) %in% drops)]

# saveRDS(test_num, "test_num.rds")
# saveRDS(train_num, "train_num.rds")

gc()

#-------------------------------------#
# note: leave-one-fold-out encoding (feifeiyu1204@gmail correspondence)
# note: additional models will use feature hashing instead for this section
set.seed(96707)
folds <- createMultiFolds(target, k = 10, times = 1)

ohe.list <- names(train[, sapply(train, is.character)])
ohe.vars <- c("ID", "target", ohe.list)

train <- train[, (names(train) %in% ohe.vars)] # take this list and merge with above for the feature hashed set
test  <- test[, (names(test) %in% ohe.vars)]

# note: mean
for (j in ohe.list) {
  list.x <- list()

  for (i in seq(1, length(folds))) {
    lofo.ids <- folds[[i]]

    x <- train[lofo.ids, ] %>%
      group_by_(j) %>%
      summarise(v_avg = mean(target, na.rm = T))

    y <- train[-lofo.ids, ] %>% left_join(x) %>% select_("ID", j, "v_avg")
    list.x[[i]] <- y
  }
  list.x <- do.call(rbind, list.x)

  list.y <- list.x %>%
    group_by_(j) %>%
    summarise(v_avg = mean(v_avg, na.rm = T))

  names(list.x)[3] <- paste0(j,"_avg")
  train <- train %>% left_join(list.x[, c(1,3)], by = "ID")

  names(list.y)[2] <- paste0(j,"_avg")
  test <- test %>% left_join(list.y)
}
names(train)
rm(list.x, list.y, x, y, lofo.ids)

# note: N
for (j in ohe.list) {
  list.x <- list()
  
  for (i in seq(1, length(folds))) {
    lofo.ids <- folds[[i]]
    
    x <- train[lofo.ids, ] %>%
      group_by_(j) %>%
      summarise(v_n = n())
    
    y <- train[-lofo.ids, ] %>% left_join(x) %>% select_("ID", j, "v_n")
    list.x[[i]] <- y
  }
  list.x <- do.call(rbind, list.x)
  
  list.y <- list.x %>%
    group_by_(j) %>%
    summarise(v_n = mean(v_n, na.rm = T))
  
  names(list.x)[3] <- paste0(j,"v_n")
  train <- train %>% left_join(list.x[, c(1,3)], by = "ID")
  
  names(list.y)[2] <- paste0(j,"v_n")
  test <- test %>% left_join(list.y)
}
names(train)
rm(list.x, list.y, x, y, lofo.ids)

# note: SD
for (j in ohe.list) {
  list.x <- list()
  
  for (i in seq(1, length(folds))) {
    lofo.ids <- folds[[i]]
    
    x <- train[lofo.ids, ] %>%
      group_by_(j) %>%
      summarise(v_sd = sd(target, na.rm = T))
    
    y <- train[-lofo.ids, ] %>% left_join(x) %>% select_("ID", j, "v_sd")
    list.x[[i]] <- y
  }
  list.x <- do.call(rbind, list.x)
  
  list.y <- list.x %>%
    group_by_(j) %>%
    summarise(v_sd = mean(v_sd, na.rm = T))
  
  names(list.x)[3] <- paste0(j,"v_sd")
  train <- train %>% left_join(list.x[, c(1,3)], by = "ID")
  
  names(list.y)[2] <- paste0(j,"v_sd")
  test <- test %>% left_join(list.y)
}
names(train)
rm(list.x, list.y, x, y, lofo.ids)
# saveRDS(test, "test.rds")
# saveRDS(train, "train.rds")

#-------------------------------------#
# note: concatenate train/test set

# test  <- readRDS("test.rds")
# train <- readRDS("train.rds")
# test_num  <- readRDS("test_num.rds")
# train_num <- readRDS("train_num.rds")

train_df <- cbind(train, train_fs)
test_df  <- cbind(test, test_fs)

# note: fix for numeric encoding
train_df <- train_df[, !(names(train_df) %in% ohe.list)]
test_df  <- test_df[, !(names(test_df) %in% ohe.list)]

train_df <- train_df %>% replace(is.na(.), -99999)
test_df <- test_df %>% replace(is.na(.), -99999)

sum(is.na(train_df))
rm(test_num, train_num, train, test)
gc()

train_df <- train_df %>% subset(., select = -c(ID, target))
test_df  <- test_df %>% subset(., select = -c(ID, target))

#=====================================#
# model data
LogLoss.score <- function(actual, predicted) {
  ll <- -1 / length(actual) * (sum((actual * log(predicted) + (1 - actual) * log(1 - predicted))))
  return(ll)
}

LogLoss.eval <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  retval <- LogLoss.score(labels, preds)
  return (list(metric = "LogLoss", value = retval))
}

# note: split set
set.seed(96707)
part <- createDataPartition(target, p = .8, list = F, times = 1)

test_DM <- xgb.DMatrix(data = data.matrix(test_df))

xgtrain <- xgb.DMatrix(data = data.matrix(train_df[part, ]), label = target[part])
xgval   <- xgb.DMatrix(data = data.matrix(train_df[-part, ]), label = target[-part])
watchlist <- list(val = xgval, train = xgtrain)

#--------------------------------------
# tuning
# note: adapted grid search from https://github.com/szilard/benchm-ml/blob/master/3-boosting/0-xgboost-init-grid.R

tune_grid <- expand.grid(eta = c(0.3),
                         max_depth = c(3, 5, 7, 9) , #4step 1: c(3, 5, 7, 9) # step 2: c(-1,0,+1)
                         min_child_weight = c(1, 3, 5), #step 1: c(1, 3, 5) # step 2: c(-1,0,+1)
                         gamma = c(0.0), # step3: c(0.0, 0.1, 0.2, 0.3, 0.4)
                         subsample =  c(0.95), #  step4: c(0.6,0.7,0.8,0.9) # step5: c(-0.05, 0, +0.05)
                         colsample_bytree = c(0.5), # step4: c(0.6,0.7,0.8,0.9) # step5: c(-0.05, 0, +0.05)
                         alpha = c(0), #c(1e-5, 1e-2,0, 0.1, 1, 100)
                         scale_pos_weight = 1 # 1 if class imbalance
)

my_list <- list()
for (k in 1:nrow(tune_grid)) {
  prm <- tune_grid[k,]
  print(prm)
  print(system.time({
    n_proc <- 4
    set.seed(2016)
    md <- xgb.cv(data = xgtrain,
                 nthread = n_proc,
                 objective = "binary:logistic",
                 nround = 75,
                 max_depth = prm$max_depth,
                 eta = prm$eta,
                 min_child_weight = prm$min_child_weight,
                 subsample = prm$subsample,
                 eval_metric = "logloss",
                 early_stop_round = 25,
                 nfold = 5,
                 maximize = F)
    my_list[[k]] <- data.frame(prm, iter = which(md$test.logloss.mean == min(md$test.logloss.mean)), score = min(md$test.logloss.mean))
  }))
}

grid_final <- do.call(rbind, my_list)
arrange(grid_final, score, iter)

# 0.469817
# 0.466785 - only lofo encoding
# 0.465986+0.004291
# 0.465829+0.004147
#--------------------------------------
# create cv
list.layers <- names(rev(sort(table(target))))

num.folds <- 2
num.repeats <- 2
num.rounds <- num.folds * num.repeats

set.seed(96707)
folds <- createMultiFolds(target, k = num.folds, times = num.repeats)

xgval.pred  <- matrix(0, nrow = nrow(train_df), ncol = 1)
xgval.rank  <- matrix(0, nrow = nrow(train_df), ncol = 1)
xgtest.pred <- matrix(0, nrow = nrow(test_df), ncol = 1)
xgtest.rank <- matrix(0, nrow = nrow(test_df), ncol = 1)

xgtest <- xgb.DMatrix(data = data.matrix(test_df))

bestIters <- rep(0, length(folds))

for (i in seq(1, length(folds)) ) {

  val.ids <- setdiff(1:nrow(train_df), folds[[i]])

  print(length(val.ids))
  print(names(folds)[i])

  train_DM  <- xgb.DMatrix(data = data.matrix(train_df[-val.ids,]),
                           label = target[-val.ids])

  val_DM    <- xgb.DMatrix(data = data.matrix(train_df[val.ids,]),
                           label = target[val.ids])

  watch_list <- list(val = val_DM, train = train_DM)

  # param <- list( max.depth = 9, eta = 0.05, booster = "gbtree",
  #                subsample = 1.0, colsample_bytree = 0.4, min_child_weight = 2,
  #                objective = "binary:logistic", eval_metric = "logloss")

  param <- list(eta = 0.01,
                max.depth = 5, subsample = 0.95,
                min_child_weight = 1,
                colsample_bytree = 0.5,
                booster = "gbtree",
                scale_pos_weight = 1,
                objective = "binary:logistic",
                gamma = 0,
                eval_metric = "logloss")

  set.seed(123)
  model1 <- xgb.train(params = param, data = train_DM, nrounds = 10000,
                      early.stop.round = 300,
                      nthread = 4, verbose = 1, print.every.n = 2,
                      watchlist = watch_list, maximize = F)

  bestIter <- model1$bestInd
  bestIters[i] <- bestIter

  xgval.pred.fold <- predict(model1, newdata = val_DM, ntreelimit = bestIter)
  xgval.pred[val.ids,] <- xgval.pred[val.ids,] + xgval.pred.fold

  pred1 <- data.frame(xgval.pred.fold)

  print(paste("prob based = ", LogLoss.score(target[val.ids], pred1)))

  xgtest.pred.fold <- predict(model1, newdata = test_DM, ntreelimit = bestIter)
  xgtest.pred <- xgtest.pred + xgtest.pred.fold
}

xgtest.pred.fn <- xgtest.pred / num.rounds
xgval.pred.fn  <- xgval.pred / num.repeats

print(bestIters)

#--------------------------------------
# evaluate cv
predictions <- data.frame(xgval.pred.fn)
print(LogLoss.score(target, predictions))
# 0.464983

mdl.xgcv <- xgb.dump(model1, with.stats=TRUE)
importance_matrix <- xgb.importance(dimnames(train_df)[[2]], model=model1)
gp <- xgb.plot.importance(importance_matrix)
print(gp)
importance_matrix[1:20]
write_csv(importance_matrix, "important_features.csv")
#=====================================#
# PREPARE SUBMISSION

#--------------------------------------
# create submission
predictions <- data.frame(xgtest.pred.fn)
predictions <- as.vector(predictions)

submission <- NULL
idx <- test$ID
id_mtx <-  matrix(idx, 1)[rep(1,1), ]
ids <- c(id_mtx)
submission$id <- ids

submission$PredictedProb <- data.frame(xgtest.pred.fn)
submission <- as.data.frame(submission)
head(submission)
names(submission) <- c("id", "PredictedProb")

write.csv(submission, "submission_with_cv2.csv", quote=FALSE, row.names = FALSE)
##  0.45832


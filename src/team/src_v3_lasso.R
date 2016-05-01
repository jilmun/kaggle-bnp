# bmp paribas cardis

#--------------------------------------
# dependencies
source("https://git.io/vz9AS")
pacman::p_load(caret, Matrix, xgboost)
setwd("/home/tom/Dropbox/bnp_paribas_cardif/")

# load data
tmp <- tempdir()
dir.files <- list.files("data")

for (i in dir.files) {
  unzip(paste("data", i, sep = "/"), exdir = tmp)
}

dir.tmp  <- list.files(tmp, pattern = ".csv", full.names = T)
list.tmp <- lapply(dir.tmp, read_csv)

names(list.tmp) <- str_replace_all(list.files(tmp, pattern = ".csv"), ".csv", "")

invisible(lapply(names(list.tmp), function(x)
  assign(x, list.tmp[[x]], envir = .GlobalEnv)))

gc()

LogLoss.score <- function(actual, predicted) {
  ll <- -1 / length(actual) * (sum((actual * log(predicted) +
                                      (1 - actual) * log(1 - predicted))))
  return(ll)
}

#--------------------------------------
# preprocess missing characteristics
target <- train$target

full <- rbind_list(train, test) %>%
  select(-ID, -target) %>%
  map_if(is.character, as.factor) %>% 
  map_if(is.factor, as.numeric) %>% 
  data.frame %>% 
  mutate(row_sum = rowSums(., na.rm = T),
         N = ncol(.), 
         row_miss_ratio = rowSums(is.na(.)) / N,
         row_var = apply(., 1, var, na.rm = T),
         row_miss_N = rowSums(is.na(.)), 
         row_zeroes = rowSums(. == 0, na.rm = T) / N,
         row_below0 = rowSums(. < 0, na.rm = T) / N
         ) %>%
  select(-N)

#--------------------------------------
# preprocess for layering
na.roughfix2 <- function (object, ...) {
  res <- lapply(object, roughfix)
  structure(res, class = "data.frame", row.names = seq_len(nrow(object)))
}

roughfix <- function(x) {
  missing <- is.na(x)
  if (!any(missing)) return(x)
  
  if (is.numeric(x)) {
    x[missing] <- median.default(x[!missing])
  } else if (is.factor(x)) {
    freq <- table(x)
    x[missing] <- names(freq)[which.max(freq)]
  } else {
    stop("na.roughfix only works for numeric or factor")
  }
  x
}

train_df <- full[(1:nrow(train)),]
test_df <- full[(1+nrow(train)):nrow(full),]

train_stx <- na.roughfix2(train_df[,feats])
test_stx <- na.roughfix2(test_df[,feats])

# create layer-1: lasso regression
pacman::p_load(doParallel)
pacman::p_load(glmnet)

fn_logit <- function(mtrain) {
  registerDoParallel(4)
  xmat <- model.matrix(target~., data=mtrain)[,-1]
  mfit <- cv.glmnet(xmat, as.factor(mtrain$target), family="binomial",
                    type.measure="class", alpha=1, parallel = T)  # alpha: 0=ridge, 1=lasso
  return(mfit)
}

clf1 <- fn_logit(cbind(target, train_stx))

plot(clf1 )
(lambda <- clf1 $lambda.min)  # best lambda with cv

clf1_train <- predict(clf1, s=lambda, newx=as.matrix(train_stx), type="response")
clf1_test <- predict(clf1, s=lambda, newx=as.matrix(test_stx), type="response")

LogLoss.score(target, clf1_train) # train ll = 0.502 (0.4937264)

#--------------------------------------
# simple stacking
# train_stx, test_stx, train_pp, test_pp

train_pp <- train_df %>% replace(is.na(.), -1)
test_pp  <- test_df %>% replace(is.na(.), -1)

train_pp$clf1_train <- clf1_train
test_pp$clf1_test <- clf1_test

train_stx$clf1_train <- clf1_train
test_stx$clf1_test <- clf1_test 

# split imputed set
set.seed(96707)
part <- createDataPartition(target, p = .75, list = F, times = 1)

xgtest_stx <- xgb.DMatrix(data = data.matrix(test_stx))

xgtrain_stx <- xgb.DMatrix(data = data.matrix(train_stx[part, ]), label = target[part])
xgval_stx <- xgb.DMatrix(data = data.matrix(train_stx[-part, ]), label = target[-part])

xgtest_pp  <- xgb.DMatrix(data = data.matrix(test_pp), missing = -1)

xgtrain_pp <- xgb.DMatrix(data = data.matrix(train_pp[part, ]), label = target[part])
xgval_pp <- xgb.DMatrix(data = data.matrix(train_pp[-part, ]), label = target[-part])

docv <- function(xgtrain, param0, iter) {
  model_cv = xgb.cv(
    params = param0, 
    nrounds = iter, 
    nfold = 5, 
    data = xgtrain, 
    early.stop.round = 15,
    maximize = FALSE, 
    nthread = 4
  )
  gc()
  
  best <- min(model_cv$test.logloss.mean)
  bestIter <- which(model_cv$test.logloss.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1
}

doTest <- function(xgtrain, xgval, param0, iter) {
  watchlist <- list(val = xgval, train = xgtrain)
    model = xgb.train(nrounds = iter, 
                    params = param0, 
                    data = xgtrain, 
                    watchlist = watchlist, 
                    print.every.n = 20,
                    nthread = 4)
  p <- predict(model, xgtest)
  rm(model)
  gc()
  p
}

param0 <- list(
  "objective"  = "binary:logistic", 
  "eval_metric" = "logloss",
  "eta" = 0.05,
  "subsample" = 0.9,
  "colsample_bytree" = 0.9,
  "min_child_weight" = 1,
  "max_depth" = 10
)

set.seed(96707)
cv_stx <- docv(xgtrain_stx, param0, 500) 
cv_pp  <- docv(xgtrain_pp, param0, 500) 

# original 1:           0.330583          0.002401          0.468019         0.001848
# 3 layers 1:           0.412849          0.001916          0.468209         0.007826
# 2 layers 1:           0.421028          0.002173          0.468522          0.00823

# revisiting simpler model
# imputed  1:           0.323564          0.001718          0.467323         0.004452
# missing  1:           0.318693          0.003917          0.467908         0.003465

ensemble_stx <- rep(0, nrow(test))
ensemble_pp  <- rep(0, nrow(test))

cv_stx2 <- round(cv_stx * 1.25)
cv_pp2  <- round(cv_pp * 1.25)

for (i in 1:10) {
  print(i)
  set.seed(i + 96707)
  p_stx <- doTest(xgtrain_stx, xgval_stx, param0, cv_stx2) 
  ensemble_stx <- ensemble_stx + p_stx
}

for (i in 1:10) {
  print(i)
  set.seed(i + 96707)
  p_pp  <- doTest(xgtrain_pp, xgval_pp, param0, cv_pp2) 
  ensemble_pp <- ensemble_pp + p_pp
}

# combine by averaging
pred1 = ensemble_stx / i
pred2 = ensemble_pp / i
PredictedProb <- (pred1 + pred2) / 2

submission <- NULL
idx <- test$ID
id_mtx <-  matrix(idx, 1)[rep(1,1), ]
ids <- c(id_mtx)
submission$id <- ids

submission$PredictedProb <- nsemble_stx / i
submission <- as.data.frame(submission)
head(submission)

write.csv(submission, "submission_raw.csv", quote=FALSE, row.names = FALSE)
# 0.88527
1:           0.412849          0.001916          0.468209         0.007826
Your submission scored 0.46586, which is not an improvement of your best score. Keep trying! 
# Import library
library(tidyverse)
library(caret)
set.seed(3759274)


## create training and test set
#full data
scouting <- read.csv("scouting_stats_apm.csv") %>% 
  mutate(Win = ifelse(Win == 1, "win", "loss"))

scouting$Win <- factor(scouting$Win, levels = c("win", "loss"))
n <- nrow(scouting)
train_index <- sample(n, round(.8*n))
scouting_train <- scouting %>% slice(train_index)
scouting_test <- scouting %>% slice(-train_index)

#sample data
scouting_sample <- read.csv("scouting_stats_apm_sample.csv") %>% 
  mutate(Win = ifelse(Win == 1, "win", "loss"))

scouting_sample$Win <- factor(scouting_sample$Win, levels = c("win", "loss"))
n <- nrow(scouting_sample)
train_index_sample <- sample(n, round(.8*n))
scouting_train_sample <- scouting_sample %>% slice(train_index_sample)
scouting_test_sample <- scouting_sample %>% slice(-train_index_sample)

train_control <- trainControl(
  method = "cv", # cross-validation
  number = 10, # 10-folds
)


#========================== KNN ==========================# 

scouting_knn <- train(
  Win ~ APM,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn <- caret::confusionMatrix(
  data = predict(scouting_knn, newdata = scouting_test),
  reference = scouting_test$Win,
  positive = "win")

## => KNN with just APM has 54% accuracy

scouting_knn_rel <- train(
  Win ~ APM + RelAPM,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn_rel <- caret::confusionMatrix(
  data = predict(scouting_knn_rel, newdata = scouting_test),
  reference = scouting_test$Win,
  positive = "win")

## => KNN with APM and RelAPM has 62.7% accuracy


#========================== Random Forrest ==========================# 
mtry_grid <- data.frame(mtry = seq(1, 1, by = 1))
scouting_rf_cv <- train(
  Win ~ APM,
  data = scouting_train_sample, # training data
  method = "rf", # classification method
  tuneGrid = mtry_grid, # rf parameters
  trControl = train_control # validation method
)

confm_test_rf <- caret::confusionMatrix(
  data = predict(scouting_rf_cv, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Random forest with just APM has 50.3% accuracy

scouting_rf_cv_rel <- train(
  Win ~ APM + RelAPM,
  data = scouting_train_sample, # training data
  method = "rf", # classification method
  tuneGrid = mtry_grid, # rf parameters
  trControl = train_control # validation method
)
confm_test_rf_rel <- caret::confusionMatrix(
  data = predict(scouting_rf_cv_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Random forest with APM and RelAPM has 60.8% accuracy

#========================== Logistic Regression ==========================# 
scouting_logit <- train(
  Win ~ APM,
  data = scouting_train, # training data
  method = "glm",
  family = "binomial",
  trControl = train_control # validation method
)
confm_test_logit <- caret::confusionMatrix(
  data = predict(scouting_logit, newdata = scouting_test),
  reference = scouting_test$Win,
  positive = "win")

## => Logistic regression with just APM has 55.5% accuracy

scouting_logit_rel <- train(
  Win ~ APM + RelAPM,
  data = scouting_train, # training data
  method = "glm",
  family = "binomial",
  trControl = train_control # validation method
)
confm_test_logit_rel <- caret::confusionMatrix(
  data = predict(scouting_logit_rel, newdata = scouting_test),
  reference = scouting_test$Win,
  positive = "win")

## => Logistic regression with APM and RelAPM has 66.4% accuracy

#========================== Stochastic Gradient Boosting Machine ==========================# 
scouting_gbm <- train(
  Win ~ APM, 
  data = scouting_train,
  method = "gbm",
  trControl = train_control,
  verbose = FALSE
)  

confm_test_gbm <- caret::confusionMatrix(
  data = predict(scouting_gbm, newdata = scouting_test),
  reference = scouting_test$Win,
  positive = "win")

## => Gradient boosting machine with just APM has 57.7% accuracy

scouting_gbm_rel <- train(
  Win ~ APM + RelAPM, 
  data = scouting_train,
  method = "gbm",
  trControl = train_control,
  verbose = FALSE
)  

confm_test_gbm_rel <- caret::confusionMatrix(
  data = predict(scouting_gbm_rel, newdata = scouting_test),
  reference = scouting_test$Win,
  positive = "win")

## => Gradient boosting machine with APM and RelAPM has 66.3% accuracy


#========================== AdaBoost Classification Trees ==========================# 
scouting_adaboost <- train(
  Win ~ APM, 
  data = scouting_train_sample,
  method = "adaboost",
  trControl = train_control
)  

confm_test_adaboost <- caret::confusionMatrix(
  data = predict(scouting_adaboost, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => AdaBoost Classification Trees with just APM has 51.4% accuracy

scouting_adaboost_rel <- train(
  Win ~ APM + RelAPM, 
  data = scouting_train_sample,
  method = "adaboost",
  trControl = train_control
)  

confm_test_adaboost_rel <- caret::confusionMatrix(
  data = predict(scouting_adaboost_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => AdaBoost Classification Trees with APM and RelAPM has 61.06% accuracy


#========================== Boosted Classification Trees ==========================# 
scouting_ada <- train(
  Win ~ APM, 
  data = scouting_train_sample,
  method = "ada",
  trControl = train_control
)  

confm_test_ada <- caret::confusionMatrix(
  data = predict(scouting_ada, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Boosted Classification Trees with just APM has 42.6% accuracy


#========================== Boosted Logistic Regression ==========================# 
scouting_logitboost <- train(
  Win ~ APM, 
  data = scouting_train_sample,
  method = "LogitBoost",
  trControl = train_control
)  

confm_test_logitboost <- caret::confusionMatrix(
  data = predict(scouting_logitboost, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Boosted Classification Trees with just APM has 57.2% accuracy

scouting_logitboost_rel <- train(
  Win ~ APM + RelAPM, 
  data = scouting_train_sample,
  method = "LogitBoost",
  trControl = train_control
)  

confm_test_logitboost_rel <- caret::confusionMatrix(
  data = predict(scouting_logitboost_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Boosted Classification Trees with APM and RelAPM has 65.2% accuracy


#========================== Model Averaged Neural Network ==========================# 
scouting_avnn <- train(
  Win ~ APM, 
  data = scouting_train_sample,
  method = "avNNet",
  trControl = train_control
)  

confm_test_avnn <- caret::confusionMatrix(
  data = predict(scouting_avnn, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Model Averaged Neural Network with just APM has 56.8% accuracy

scouting_avnn_rel <- train(
  Win ~ APM + RelAPM, 
  data = scouting_train_sample,
  method = "avNNet",
  trControl = train_control
)  

confm_test_avnn_rel <- caret::confusionMatrix(
  data = predict(scouting_avnn_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Model Averaged Neural Network with APM and RelAPM has 66.4% accuracy


#========================== Random Ferns ==========================# 
scouting_rfern <- train(
  Win ~ APM, 
  data = scouting_train_sample,
  method = "rFerns",
  trControl = train_control
)  

confm_test_rfern <- caret::confusionMatrix(
  data = predict(scouting_rfern, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Random ferns with just APM has 57.1% accuracy

scouting_rfern_rel <- train(
  Win ~ APM + RelAPM, 
  data = scouting_train_sample,
  method = "rFerns",
  trControl = train_control
)  

confm_test_rfern_rel <- caret::confusionMatrix(
  data = predict(scouting_rfern_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Random ferns with APM and RelAPM has 67.5% accuracy


#========================== Bagged CART ==========================# 
scouting_treebag <- train(
  Win ~ APM, 
  data = scouting_train_sample,
  method = "treebag",
  trControl = train_control
)  

confm_test_treebag <- caret::confusionMatrix(
  data = predict(scouting_treebag, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Bagged CART with just APM has 53.9% accuracy

scouting_treebag_rel <- train(
  Win ~ APM + RelAPM, 
  data = scouting_train_sample,
  method = "treebag",
  trControl = train_control
)  

confm_test_treebag_rel <- caret::confusionMatrix(
  data = predict(scouting_treebag_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Win,
  positive = "win")

## => Bagged CART with APM and RelAPM has 60.1% accuracy



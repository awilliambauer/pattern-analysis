# Import library
library(tidyverse)
library(caret)
library(forcats)
set.seed(3759274)


## create training and test set

#full data
scouting <- read.csv("scouting_stats_apm.csv") %>% 
  mutate(Win = ifelse(Win == 1, "win", "loss")) %>% 
  filter(Rank %in% c(1, 2, 4, 5, 7))
scouting$Win <- factor(scouting$Win, levels = c("win", "loss"))
scouting$Rank <- as.factor(scouting$Rank)
scouting <- scouting %>% 
  mutate(Skill = fct_collapse(Rank, 
                              Novice = c("1", "2"),
                              Proficient = c("4", "5"),
                              Expert = c("7")))
n <- nrow(scouting)
train_index <- sample(n, round(.8*n))
scouting_train <- scouting %>% slice(train_index)
scouting_test <- scouting %>% slice(-train_index)

#sample data
scouting_sample <- read.csv("scouting_stats_apm_sample.csv") %>% 
  mutate(Win = ifelse(Win == 1, "win", "loss")) %>% 
  filter(Rank %in% c(1, 2, 4, 5, 7))

scouting_sample$Win <- factor(scouting_sample$Win, levels = c("win", "loss"))
scouting_sample$Rank <- as.factor(scouting_sample$Rank)
scouting_sample <- scouting_sample %>% 
  mutate(Skill = fct_collapse(Rank, 
                              Novice = c("1", "2"),
                              Proficient = c("4", "5"),
                              Expert = c("7")))
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
  Skill ~ APM,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn <- caret::confusionMatrix(
  data = predict(scouting_knn, newdata = scouting_test),
  reference = scouting_test$Skill)

## => KNN with just APM has 84% accuracy

scouting_knn_rel <- train(
  Skill ~ APM + RelAPM,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn_rel <- caret::confusionMatrix(
  data = predict(scouting_knn_rel, newdata = scouting_test),
  reference = scouting_test$Skill)

## => KNN with APM and RelAPM has 86.9% accuracy


#========================== Random Forrest ==========================# 
mtry_grid <- data.frame(mtry = seq(1, 1, by = 1))
scouting_rf_cv <- train(
  Skill ~ APM,
  data = scouting_train_sample, # training data
  method = "rf", # classification method
  tuneGrid = mtry_grid, # rf parameters
  trControl = train_control # validation method
)

confm_test_rf <- caret::confusionMatrix(
  data = predict(scouting_rf_cv, newdata = scouting_test_sample),
  reference = scouting_test_sample$Skill)

## => Random forest with just APM has 78.8% accuracy

scouting_rf_cv_rel <- train(
  Skill ~ APM + RelAPM,
  data = scouting_train_sample, # training data
  method = "rf", # classification method
  tuneGrid = mtry_grid, # rf parameters
  trControl = train_control # validation method
)
confm_test_rf_rel <- caret::confusionMatrix(
  data = predict(scouting_rf_cv_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Skill)

## => Random forest with APM and RelAPM has 83.9% accuracy


#========================== Boosted Logistic Regression ==========================# 
scouting_logitboost <- train(
  Skill ~ APM, 
  data = scouting_train,
  method = "LogitBoost",
  trControl = train_control
)  

confm_test_logitboost <- caret::confusionMatrix(
  data = predict(scouting_logitboost, newdata = scouting_test),
  reference = scouting_test$Skill)

## => Boosted Logistic Regression with just APM has 86% accuracy

scouting_logitboost_rel <- train(
  Skill ~ APM + RelAPM, 
  data = scouting_train,
  method = "LogitBoost",
  trControl = train_control
)  

confm_test_logitboost_rel <- caret::confusionMatrix(
  data = predict(scouting_logitboost_rel, newdata = scouting_test),
  reference = scouting_test$Skill)

## => Boosted Logistic Regression with APM and RelAPM has 88.5% accuracy


#========================== Model Averaged Neural Network ==========================# 
scouting_avnn <- train(
  Skill ~ APM, 
  data = scouting_train_sample,
  method = "avNNet",
  trControl = train_control
)  

confm_test_avnn <- caret::confusionMatrix(
  data = predict(scouting_avnn, newdata = scouting_test_sample),
  reference = scouting_test_sample$Skill)

## => Model Averaged Neural Network with just APM has 83.9% accuracy

scouting_avnn_rel <- train(
  Skill ~ APM + RelAPM, 
  data = scouting_train_sample,
  method = "avNNet",
  trControl = train_control
)  

confm_test_avnn_rel <- caret::confusionMatrix(
  data = predict(scouting_avnn_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Skill)

## => Model Averaged Neural Network with APM and RelAPM has 83.6% accuracy


#========================== Bagged CART ==========================# 
scouting_treebag <- train(
  Skill ~ APM, 
  data = scouting_train_sample,
  method = "treebag",
  trControl = train_control
)  

confm_test_treebag <- caret::confusionMatrix(
  data = predict(scouting_treebag, newdata = scouting_test_sample),
  reference = scouting_test_sample$Skill)

## => Bagged CART with just APM has 79.1% accuracy

scouting_treebag_rel <- train(
  Skill ~ APM + RelAPM, 
  data = scouting_train_sample,
  method = "treebag",
  trControl = train_control
)  

confm_test_treebag_rel <- caret::confusionMatrix(
  data = predict(scouting_treebag_rel, newdata = scouting_test_sample),
  reference = scouting_test_sample$Skill)

## => Bagged CART with APM and RelAPM has 86.1% accuracy


#==========================Redo with 1 replay per player ==========================# 
# sample data with 1 replay per player (1)
scouting_sample_single_player <- scouting %>% 
  group_by(UID) %>% 
  slice_sample(n = 1) %>% 
  ungroup()

# equalize the number of players in each category (2)
scouting_sample_single_player <- scouting %>% 
  group_by(Skill) %>% 
  slice_sample(n = 500) %>% 
  ungroup()

set.seed(525701)
n <- nrow(scouting_sample_single_player)
train_index <- sample(n, round(.8*n))
scouting_train <- scouting_sample_single_player %>% slice(train_index)
scouting_test <- scouting_sample_single_player %>% slice(-train_index)


train_control <- trainControl(
  method = "cv", # cross-validation
  number = 10, # 10-folds
)

#========================== Visualization ==========================#
ggplot(scouting_sample_single_player, aes(x = Rank, y = APM)) +
  geom_boxplot() + 
  scale_y_log10()

ggplot(scouting_sample_single_player, aes(x = Skill, y = APM)) +
  geom_boxplot() +
  scale_y_log10()

#========================== KNN ==========================# 

scouting_knn <- train(
  Skill ~ APM,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn <- caret::confusionMatrix(
  data = predict(scouting_knn, newdata = scouting_test),
  reference = scouting_test$Skill)

## => KNN with just APM has 85% (1) and 71% accuracy (2)

scouting_knn_rel <- train(
  Skill ~ APM + RelAPM,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn_rel <- caret::confusionMatrix(
  data = predict(scouting_knn_rel, newdata = scouting_test),
  reference = scouting_test$Skill)

## => KNN with just APM has 88% (1) and 80% accuracy (2)

# Import library
library(tidyverse)

#sample data
scouting <- read.csv("scouting_stats_apm_sample.csv") %>% 
  mutate(Win = ifelse(Win == 1, "win", "loss")) %>% 
  filter(VarAPM != "NaN", Rank %in% c(1, 2, 4, 5, 7)) 
scouting$Rank <- as.factor(scouting$Rank)
scouting <- scouting %>% 
  mutate(Skill = fct_collapse(Rank, 
                              Novice = c("1", "2"),
                              Proficient = c("4", "5"),
                              Expert = c("7")))

# Export data with outliers
scouting_outlier <- scouting %>% 
  filter(UnusualAPMatMin != "")
write.csv(scouting_outlier, "scouting_outlier.csv")

set.seed(525701)
n <- nrow(scouting)
train_index <- sample(n, round(.8*n))
scouting_train <- scouting %>% slice(train_index)
scouting_test <- scouting %>% slice(-train_index)


train_control <- trainControl(
  method = "cv", # cross-validation
  number = 10, # 10-folds
)


#========================== Visualization ==========================#
ggplot(scouting, aes(x = Rank, y = VarAPM)) +
  geom_boxplot() + 
  scale_y_log10()

ggplot(scouting, aes(x = Skill, y = VarAPM)) +
  geom_boxplot() +
  scale_y_log10()

#========================== KNN ==========================# 

scouting_knn <- train(
  Skill ~ VarAPM,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn <- caret::confusionMatrix(
  data = predict(scouting_knn, newdata = scouting_test),
  reference = scouting_test$Skill)

## => KNN with just APM has 84% accuracy


#========================== Boosted Logistic Regression ==========================# 
scouting_logitboost <- train(
  Skill ~ VarAPM, 
  data = scouting_train,
  method = "LogitBoost",
  trControl = train_control
)  

confm_test_logitboost <- caret::confusionMatrix(
  data = predict(scouting_logitboost, newdata = scouting_test),
  reference = scouting_test$Skill)

## => Boosted Logistic Regression with just APM has 84.9% accuracy

#========================== Model Averaged Neural Network ==========================# 
scouting_avnn <- train(
  Skill ~ VarAPM, 
  data = scouting_train,
  method = "avNNet",
  trControl = train_control
)  

confm_test_avnn <- caret::confusionMatrix(
  data = predict(scouting_avnn, newdata = scouting_test),
  reference = scouting_test$Skill)

## => Model Averaged Neural Network with just APM has 84.9% accuracy



#==========================Redo with 1 replay per player ==========================# 
# sample data with 1 replay per player
scouting_sample_single_player <- scouting %>% 
  group_by(UID) %>% 
  slice_sample(n = 1) %>% 
  ungroup()

# equalize the number of players in each category
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
ggplot(scouting_sample_single_player, aes(x = Rank, y = VarAPM)) +
  geom_boxplot() + 
  scale_y_log10()

ggplot(scouting_sample_single_player, aes(x = Skill, y = VarAPM)) +
  geom_boxplot() +
  scale_y_log10()

#========================== KNN ==========================# 

scouting_knn <- train(
  Skill ~ VarAPM,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn <- caret::confusionMatrix(
  data = predict(scouting_knn, newdata = scouting_test),
  reference = scouting_test$Skill)

## => KNN with just APM has 59% accuracy with the 
## first sample and 83% with the second sample
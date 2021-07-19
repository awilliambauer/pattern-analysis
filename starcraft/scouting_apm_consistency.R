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

scouting_player <- scouting %>% 
  group_by(UID) %>% 
  summarise(num = n())

#========================== Modal skill ~ VarAPM_UID ==========================# 

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

scouting_apm_by_player <- scouting %>% 
  group_by(UID) %>% 
  summarise(VarAPM_UID = var(APM, na.rm = T),
            ModalSkill = Mode(Skill)) %>% 
  filter(!is.na(VarAPM_UID) & VarAPM_UID != 0)

set.seed(525701)
n <- nrow(scouting_apm_by_player)
train_index_sample <- sample(n, round(.8*n))
scouting_train <- scouting_apm_by_player %>% slice(train_index_sample)
scouting_test <- scouting_apm_by_player %>% slice(-train_index_sample)

train_control <- trainControl(
  method = "cv", # cross-validation
  number = 10, # 10-folds
)

#========================== Visualization ==========================#
ggplot(scouting_apm_by_player, aes(x = ModalSkill, y = VarAPM_UID)) +
  geom_boxplot() +
  scale_y_log10()

#========================== KNN ==========================# 

scouting_knn <- train(
  ModalSkill ~ VarAPM_UID,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn <- caret::confusionMatrix(
  data = predict(scouting_knn, newdata = scouting_test),
  reference = scouting_test$ModalSkill)

## => KNN has 89.3% accuracy


#========================== Model skill ~ Number of matches ==========================# 
scouting_apm_by_player_num_match <- scouting %>% 
  group_by(UID) %>% 
  summarise(NumMatch = n(),
            ModalSkill = Mode(Skill)) %>% 
  filter(UID != 0)
n <- nrow(scouting_apm_by_player_num_match)
train_index_sample <- sample(n, round(.8*n))
scouting_train <- scouting_apm_by_player_num_match %>% slice(train_index_sample)
scouting_test <- scouting_apm_by_player_num_match %>% slice(-train_index_sample)

#========================== Visualization ==========================#
ggplot(scouting_apm_by_player_num_match, aes(x = ModalSkill, y = NumMatch)) +
  geom_boxplot() +
  scale_y_log10()

#========================== Model Averaged Neural Network ==========================# 
scouting_avnn <- train(
  ModalSkill ~ NumMatch, 
  data = scouting_train,
  method = "avNNet",
  trControl = train_control
)  

confm_test_avnn <- caret::confusionMatrix(
  data = predict(scouting_avnn, newdata = scouting_test),
  reference = scouting_test$ModalSkill)

## => Model Averaged Neural Network with just APM has 82.7% accuracy



#========================== ZeroR Evaluation ==========================# 

library(OneR)
ZeroR <- function(x, ...) {
  output <- OneR(cbind(dummy = TRUE, x[ncol(x)]), ...)
  class(output) <- c("ZeroR", "OneR")
  output
}
predict.ZeroR <- function(object, newdata, ...) {
  class(object) <- "OneR"
  predict(object, cbind(dummy = TRUE, newdata[ncol(newdata)]), ...)
}

scouting_zeroR <- scouting %>% 
  group_by(UID) %>% 
  summarise(NumMatch = n(),
            ModalSkill = Mode(Skill)) %>% 
  filter(UID != 0) %>% 
  mutate(ModalSkillNew = ifelse(ModalSkill == "Novice", 1, 
                           ifelse(ModalSkill == "Proficient", 2, 3))) %>% 
  select(-ModalSkill)
set.seed(525701)
n <- nrow(scouting_zeroR)
train_index_sample <- sample(n, round(.8*n))
scouting_train <- scouting_zeroR %>% slice(train_index_sample)
scouting_test <- scouting_zeroR %>% slice(-train_index_sample)

model <- ZeroR(scouting_train)
summary(model)
plot(model)
prediction <- predict(model, scouting_test)
eval_model(prediction, scouting_test$ModalSkillNew)

#========================== Modal skill Redo ==========================# 
set.seed(525701)
scouting_apm_by_player_sample <- scouting_apm_by_player %>% 
  group_by(ModalSkill) %>% 
  slice_sample(n = 150) %>% 
  ungroup()

set.seed(525701)
n <- nrow(scouting_apm_by_player_sample)
train_index_sample <- sample(n, round(.8*n))
scouting_train <- scouting_apm_by_player_sample %>% slice(train_index_sample)
scouting_test <- scouting_apm_by_player_sample %>% slice(-train_index_sample)

model <- ZeroR(scouting_train)
summary(model)
plot(model)
prediction <- predict(model, scouting_test)
eval_model(prediction, scouting_test$ModalSkill)

#========================== Visualization ==========================#
ggplot(scouting_apm_by_player_sample, aes(x = ModalSkill, y = VarAPM_UID)) +
  geom_boxplot() +
  scale_y_log10()

#========================== KNN ==========================# 

scouting_knn <- train(
  ModalSkill ~ VarAPM_UID,
  data = scouting_train, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn <- caret::confusionMatrix(
  data = predict(scouting_knn, newdata = scouting_test),
  reference = scouting_test$ModalSkill)

## => KNN after being resampled has 45.6% accuracy



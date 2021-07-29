# Import library
library(tidyverse)
library(caret)
library(forcats)
library(randomForest)

# Import data - League
fulldata <- read.csv("scouting_stats_cluster.csv") %>% 
  filter(!is.na(Rank)) %>% 
  mutate(Win = ifelse(Win == 1, "win", "loss"))
fulldata$Win <- factor(fulldata$Win, levels = c("win", "loss"))

# 1 rank per player
fulldata$Rank <- as.factor(fulldata$Rank)
fulldata_league <- fulldata %>% 
  mutate(League = fct_recode(Rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7")) 

  # group_by(UID) %>% 
  # filter(n_distinct(GameID) == 1) %>% 
  # ungroup()

# 1 replay per player
set.seed(11111)
fulldata_league <- fulldata_league %>% 
  group_by(UID) %>% 
  slice_sample(n = 1) %>% 
  ungroup()

set.seed(525701)
bronze_UID <- (fulldata_league %>% group_by(League) %>% 
                 filter(League == "Bronze") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 400))$UID
silver_UID <- (fulldata_league %>% group_by(League) %>% 
                 filter(League == "Silver") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 400))$UID
gold_UID <- (fulldata_league %>% group_by(League) %>% 
               filter(League == "Gold") %>% 
               distinct(UID) %>% 
               slice_sample(n = 400))$UID
platinum_UID <- (fulldata_league %>% group_by(League) %>% 
                   filter(League == "Platinum") %>% 
                   distinct(UID) %>% 
                   slice_sample(n = 400))$UID
diamond_UID <- (fulldata_league %>% group_by(League) %>% 
                  filter(League == "Diamond") %>% 
                  distinct(UID) %>% 
                  slice_sample(n = 400))$UID
master_UID <- (fulldata_league %>% group_by(League) %>% 
                 filter(League == "Master") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 400))$UID
grandmaster_UID <- (fulldata_league %>% group_by(League) %>% 
                      filter(League == "Grandmaster") %>% 
                      distinct(UID) %>% 
                      slice_sample(n = 400))$UID

uids_league <- unique(c(bronze_UID, silver_UID, gold_UID, platinum_UID, diamond_UID, 
                 master_UID, grandmaster_UID))

sampledata_league <- fulldata_league %>% 
  filter(UID %in% uids_league)

# Create train and test set
set.seed(525703)
n <- nrow(sampledata_league)
train_index_league <- sample(n, round(.8*n))
train_set_league <- sampledata_league %>% slice(train_index_league)
test_set_league <- sampledata_league %>% slice(-train_index_league)

train_control <- trainControl(
  method = "cv", # cross-validation
  number = 10, # 10-folds
)

# Data - Skill
uids_skill <- unique(c(bronze_UID, silver_UID, platinum_UID, diamond_UID, grandmaster_UID))
sampledata_skill <- fulldata_league %>% 
  filter(Rank %in% c(1, 2, 4, 5, 7),
         UID %in% uids_skill) 
sampledata_skill$Rank <- as.numeric(sampledata_skill$Rank)
sampledata_skill$Rank <- as.factor(sampledata_skill$Rank)
sampledata_skill <- sampledata_skill %>% 
  mutate(Skill = fct_collapse(Rank, 
                              Novice = c("1", "2"),
                              Proficient = c("4", "5"),
                              Expert = c("7"))) 

# Create train and test set
set.seed(525702)
n <- nrow(sampledata_skill)
train_index_skill <- sample(n, round(.8*n))
train_set_skill <- sampledata_skill %>% slice(train_index_skill)
test_set_skill <- sampledata_skill %>% slice(-train_index_skill)

#========================== KNN ==========================# 

knn_model_league <- train(
  League ~ APM + ScoutingFrequency + CPS,
  data = train_set_league, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn_league <- caret::confusionMatrix(
  data = predict(knn_model_league, newdata = test_set_league),
  reference = test_set_league$League)

## => KNN with just APM has 84% accuracy

knn_model_skill <- train(
  Skill ~ ScoutingFrequency + CPS + APM,
  data = train_set_skill, # training data
  method ="knn", # classification method
  preProc = c("center", "scale"), # standardization
  trControl = train_control # validation method
)

confm_test_knn_skill <- caret::confusionMatrix(
  data = predict(knn_model_skill, newdata = test_set_skill),
  reference = test_set_skill$Skill)

## => KNN with just APM has 90% accuracy, just ScoutingFrequency 67%, just CPS 
## 90%

cor(train_set_skill$CPS, train_set_skill$APM)

# visualization
ggplot(sampledata_skill, aes(y = Scout)) +
  geom_histogram()

#========================== Random Forrest ==========================# 
# mtry_grid <- data.frame(mtry = seq(1, 1, by = 1))
rf_model_skill <- train(
  Skill ~ APM + ScoutingFrequency + CPS,
  data = train_set_skill, # training data
  method = "rf", # classification method
  # tuneGrid = mtry_grid, # rf parameters
  trControl = train_control # validation method
)

confm_test_rf_skill <- caret::confusionMatrix(
  data = predict(rf_model_skill, newdata = test_set_skill),
  reference = test_set_skill$Skill)

rf_skill <- randomForest(Skill ~ APM + ScoutingFrequency + CPS,
                         data = train_set_skill,
                         ntree = 200)
importance(rf_skill)


#========================== Stochastic Gradient Boosting Machine ==========================# 
scouting_gbm <- train(
  Skill ~ APM + ScoutingFrequency + CPS, 
  data = train_set_skill,
  method = "xgbTree",
  trControl = train_control,
  verbose = FALSE
)  

confm_test_gbm <- caret::confusionMatrix(
  data = predict(scouting_gbm, newdata = test_set_skill),
  reference = test_set_skill$Skill)

varImp(scouting_gbm)

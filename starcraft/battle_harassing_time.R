# library
library(tidyverse)

# import data
battle_time <- read.csv("battle_time.csv") %>% 
  filter(!is.na(Rank))
harassing_time <- read.csv("harassing_time.csv") %>% 
  filter(!is.na(Rank))

# wrangle data
battle_time$Rank <- as.factor(battle_time$Rank)
battle_time <- battle_time %>% 
  mutate(League = fct_recode(Rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7"))

harassing_time$Rank <- as.factor(harassing_time$Rank)
harassing_time <- harassing_time %>% 
  mutate(League = fct_recode(Rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7"))

# ====== Visualization ===== #

ggplot(battle_time, aes(x = BattleTime)) +
  geom_histogram() +
  facet_wrap(~ League) +
  xlim(0, 2500)

ggplot(harassing_time, aes(x = HarassingTime)) +
  geom_histogram() +
  facet_wrap(~ League) +
  xlim(0, 2500)

battle_time %>% count(UID, League, GameID) %>% 
  ggplot(aes(x = League, y = n)) +
  geom_boxplot() +
  labs(title = "Frequency of battles per match by leagues")

harassing_time %>% count(UID, League, GameID) %>% 
  ggplot(aes(x = League, y = n)) +
  geom_boxplot() +
  labs(title = "Frequency of harassing per match by leagues")

battle_time %>% group_by(UID, League, GameID) %>% 
  summarise(sum_battle_duration = sum(BattleDuration)) %>% 
  ggplot(aes(x = League, y = sum_battle_duration)) +
  geom_boxplot() +
  scale_y_log10()

harassing_time %>% group_by(UID, League, GameID) %>% 
  summarise(sum_harassing_duration = sum(HarassingDuration)) %>% 
  ggplot(aes(x = League, y = sum_harassing_duration)) +
  geom_boxplot() +
  scale_y_log10()

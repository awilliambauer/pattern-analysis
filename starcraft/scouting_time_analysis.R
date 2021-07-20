# library
library(tidyverse)

# import data
scouting <- read.csv("scouting_time_seconds.csv") %>% 
  filter(!is.na(Rank))

# wrangle data
scouting$Rank <- as.factor(scouting$Rank)
scouting <- scouting %>% 
  mutate(League = fct_recode(Rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7"))

#color code
bronze_col <- "#4d4dff"
silver_col <- "#36b3b3"
gold_col <- "#884dff"
plat_col <- "#00e5e6"
diam_col <- "#3655b3"
mast_col <- "#000066"
gmast_col <- "#3c00b3"

novice_col <- "#4d4dff"
prof_col <- "#36b3b3"
expert_col <- "#884dff"

# =================== League =================== #

scouting_num_replay <- scouting %>% 
  group_by(League) %>% 
  summarise(num_replay = n_distinct(GameID) * 2)

bronze_num_replay <- (scouting_num_replay %>% filter(League == "Bronze"))[[1, 2]]
silver_num_replay <- (scouting_num_replay %>% filter(League == "Silver"))[[1, 2]]
gold_num_replay <- (scouting_num_replay %>% filter(League == "Gold"))[[1, 2]]
platinum_num_replay <- (scouting_num_replay %>% filter(League == "Platinum"))[[1, 2]]
diamond_num_replay <- (scouting_num_replay %>% filter(League == "Diamond"))[[1, 2]]
master_num_replay <- (scouting_num_replay %>% filter(League == "Master"))[[1, 2]]
gmaster_num_replay <- (scouting_num_replay %>% filter(League == "Grandmaster"))[[1, 2]]

# ===================  when =================== #

ggplot(scouting, aes(x = ScoutTime, fill = League)) +
  geom_histogram() +
  facet_wrap(~ League) +
  xlim(0, 2000) +
  labs(title = "When do players scout (unnormalized)")


bronze <- ggplot(data = subset(scouting, League == 'Bronze'), 
                 aes(x = ScoutTime, y = ..count../bronze_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Bronze") +
  xlim(0, 1500) + 
  ylim(0, 2.5)

silver <- ggplot(data = subset(scouting, League == 'Silver'), 
                 aes(x = ScoutTime, y = ..count../silver_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Silver") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)

gold <- ggplot(data = subset(scouting, League == 'Gold'), 
                 aes(x = ScoutTime, y = ..count../gold_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Gold") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)

platinum <- ggplot(data = subset(scouting, League == 'Platinum'), 
                 aes(x = ScoutTime, y = ..count../platinum_num_replay)) +
  geom_histogram(color = plat_col, fill = plat_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Platinum") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)
diamond <- ggplot(data = subset(scouting, League == 'Diamond'), 
                  aes(x = ScoutTime, y = ..count../diamond_num_replay)) +
  geom_histogram(color = diam_col, fill = diam_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Diamond") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)

master <- ggplot(data = subset(scouting, League == 'Master'), 
                 aes(x = ScoutTime, y = ..count../master_num_replay)) +
  geom_histogram(color = mast_col, fill = mast_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Master") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)

gmaster <- ggplot(data = subset(scouting, League == 'Grandmaster'), 
                 aes(x = ScoutTime, y = ..count../gmaster_num_replay)) +
  geom_histogram(color = gmast_col, fill = gmast_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Grandmaster") +
  xlim(0, 1500) + 
  ylim(0, 2.5)

grid.arrange(bronze, silver, gold, platinum, 
             diamond, master, gmaster)


# how often
scouting %>% count(UID, League, GameID) %>% 
  ggplot(aes(x = League, y = n)) +
  geom_boxplot() +
  scale_y_log10()

# =================== Race =================== #

scouting_num_replay <- scouting %>% 
  filter(League == "Grandmaster") %>% 
  group_by(Race) %>% 
  summarise(num_replay = n_distinct(GameID) * 2)

protoss_num_replay <- (scouting_num_replay %>% filter(Race == "Protoss"))[[1, 2]]  
terran_num_replay <- (scouting_num_replay %>% filter(Race == "Terran"))[[1, 2]]
zerg_num_replay <- (scouting_num_replay %>% filter(Race == "Zerg"))[[1, 2]]

scouting_gm <- scouting %>% filter(League == "Grandmaster")

protoss <- ggplot(data = subset(scouting_gm, Race == 'Protoss'), 
                 aes(x = ScoutTime, y = ..count../protoss_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = 25, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Protoss") +
  xlim(0, 2000) + 
  ylim(0, 9)

terran <- ggplot(data = subset(scouting_gm, Race == 'Terran'), 
                 aes(x = ScoutTime, y = ..count../terran_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = 25, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Terran") +
  xlim(0, 2000) + 
  ylim(0, 9)

zerg <- ggplot(data = subset(scouting, Race == 'Zerg'), 
               aes(x = ScoutTime, y = ..count../zerg_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = 25, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Zerg") +
  xlim(0, 2000) + 
  ylim(0, 9)

grid.arrange(protoss, terran, zerg)

# =========== debug ============ #

scouting_pros_terran <- scouting %>% group_by(UID, League, GameID) %>% 
  summarise(scouting_count = n(), Race = first(Race)) %>%
  group_by(Race) %>% 
  slice_min(scouting_count, n = 10)

scouting_pros_terran <- scouting %>% group_by(UID, League, GameID) %>% 
  summarise(scouting_count = n(), Race = first(Race)) %>%
  filter(Race == "Protoss" | Race == "Terran", scouting_count == 0)

scouting_pros_terran <- scouting %>% group_by(GameID) %>% 
  summarise(player_count = n_distinct(UID)) %>% 
  filter(player_count == 1)

write.csv(scouting_pros_terran, "scouting_one_player.csv")

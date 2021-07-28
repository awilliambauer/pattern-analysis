# library
library(tidyverse)

# import data
scouting <- read.csv("scouting_stats_cluster.csv") %>% 
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

# verify Zimri's data
scouting_2 <- read.csv("scouting_time_seconds_new.csv") %>% 
  filter(Rank == 7)
scouting_num_replay <- scouting_2 %>% 
  filter(Rank == 7) %>% 
  group_by(Race) %>% 
  summarise(num_replay = n_distinct(GameID) * 2)

protoss_num_replay <- (scouting_num_replay %>% filter(Race == "Protoss"))[[1, 2]]  
terran_num_replay <- (scouting_num_replay %>% filter(Race == "Terran"))[[1, 2]]
zerg_num_replay <- (scouting_num_replay %>% filter(Race == "Zerg"))[[1, 2]]

protoss <- ggplot(data = subset(scouting_2, Race == 'Protoss'), 
                  aes(x = ScoutStartTime/22.4, y = ..count../protoss_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = 25, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Protoss") +
  xlim(0, 3000) + 
  ylim(0, 1.4)

terran <- ggplot(data = subset(scouting_2, Race == 'Terran'), 
                 aes(x = ScoutStartTime/22.4, y = ..count../terran_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = 25, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Terran") +
  xlim(0, 3000) + 
  ylim(0, 1.4)

zerg <- ggplot(data = subset(scouting_2, Race == 'Zerg'), 
               aes(x = ScoutStartTime/22.4, y = ..count../zerg_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = 25, alpha = 0.6) + 
  labs(y="Scouting Instances/Replay", x = "Gametime-Zerg") +
  xlim(0, 3000) + 
  ylim(0, 1.4)

grid.arrange(protoss, terran, zerg)


# ============= verify Zimri's data ============= #

old_scouting <- read.csv("scouting_time_seconds_gm_only.csv") %>% 
  filter(Rank == 7)
new_scouting <- read.csv("scouting_instances_gm2021-07-27.csv") %>% 
  filter(!is.na(Rank1))

length(unique(old_scouting$GameID)) 
length(unique(new_scouting$GameID)) 

old_scouting_count <- old_scouting %>% count(GameID, UID)
new_scouting_count <- new_scouting %>% count(GameID, UID)

merged_count <- new_scouting_count %>% 
  inner_join(old_scouting_count, by = c("GameID", "UID")) %>% 
  mutate(diff_scouting_count = abs(n.y - n.x)) %>% 
  slice_max(diff_scouting_count, n = 20)

# duration of scouting
new_scouting$Rank1 <- as.factor(new_scouting$Rank1)
new_scouting <- new_scouting %>% 
  mutate(duration = (ScoutingEndTime - ScoutingStartTime)/22.4,
         League = fct_recode(Rank1,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7"),
         ScoutEndSec = ScoutingEndTime/22.4,
         ScoutStartSec = ScoutingStartTime/22.4)


new_scouting_player <- new_scouting %>% 
  group_by(UID1, GameId, League) %>% 
  summarise(sum_duration = sum(duration))

ggplot(new_scouting_player, aes(x = League, y = sum_duration)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(y = "Sum of scouting duration per player")


# === slow code starts here ==================================== #
new_scouting_frame <- new_scouting %>% filter(Rank == 9) %>% 
  mutate(frame = 0)
new_scouting <- new_scouting %>% mutate(frame = 0)
for(i in 1:nrow(new_scouting)) {
  temp_vec <- new_scouting[i, ]
  for(j in temp_vec[[5]]:temp_vec[[6]]) {
    cur_vec <- temp_vec
    cur_vec[[10]] <- j
    new_scouting_frame <- rbind(new_scouting_frame, cur_vec)
  }
}
# === slow code ends here ==================================== #


scouting_num_replay <- new_scouting_frame %>% 
  group_by(League) %>% 
  summarise(num_replay = n_distinct(GameID) * 2)

bronze_num_replay <- (scouting_num_replay %>% filter(League == "Bronze"))[[1, 2]]
silver_num_replay <- (scouting_num_replay %>% filter(League == "Silver"))[[1, 2]]
gold_num_replay <- (scouting_num_replay %>% filter(League == "Gold"))[[1, 2]]
platinum_num_replay <- (scouting_num_replay %>% filter(League == "Platinum"))[[1, 2]]
diamond_num_replay <- (scouting_num_replay %>% filter(League == "Diamond"))[[1, 2]]
master_num_replay <- (scouting_num_replay %>% filter(League == "Master"))[[1, 2]]
gmaster_num_replay <- (scouting_num_replay %>% filter(League == "Grandmaster"))[[1, 2]]

bronze <- ggplot(data = subset(new_scouting_frame, League == 'Bronze'), 
                 aes(x = frame/22.4, y = ..count../bronze_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = 30, alpha = 0.6) + 
  labs(y="Scouting Instances in a frame/Replay", x = "Gametime-Bronze") +
  xlim(0, 1500) + 
  ylim(0, 2.5)

silver <- ggplot(data = subset(new_scouting_frame, League == 'Silver'), 
                 aes(x = frame, y = ..count../silver_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances in a frame/Replay", x = "Gametime-Silver") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)

gold <- ggplot(data = subset(new_scouting_frame, League == 'Gold'), 
               aes(x = frame, y = ..count../gold_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances in a frame/Replay", x = "Gametime-Gold") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)

platinum <- ggplot(data = subset(new_scouting_frame, League == 'Platinum'), 
                   aes(x = frame, y = ..count../platinum_num_replay)) +
  geom_histogram(color = plat_col, fill = plat_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances in a frame/Replay", x = "Gametime-Platinum") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)
diamond <- ggplot(data = subset(new_scouting_frame, League == 'Diamond'), 
                  aes(x = frame, y = ..count../diamond_num_replay)) +
  geom_histogram(color = diam_col, fill = diam_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances in a frame/Replay", x = "Gametime-Diamond") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)

master <- ggplot(data = subset(new_scouting_frame, League == 'Master'), 
                 aes(x = frame, y = ..count../master_num_replay)) +
  geom_histogram(color = mast_col, fill = mast_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances in a frame/Replay", x = "Gametime-Master") +
  xlim(0, 1500)+ 
  ylim(0, 2.5)

gmaster <- ggplot(data = subset(new_scouting_frame, League == 'Grandmaster'), 
                  aes(x = frame, y = ..count../gmaster_num_replay)) +
  geom_histogram(color = gmast_col, fill = gmast_col,
                 bins = 40, alpha = 0.6) + 
  labs(y="Scouting Instances in a frame/Replay", x = "Gametime-Grandmaster") +
  xlim(0, 1500) + 
  ylim(0, 2.5)

grid.arrange(bronze, silver, gold, platinum, 
             diamond, master, gmaster)

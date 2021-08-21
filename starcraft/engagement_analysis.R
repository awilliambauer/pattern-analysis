# library
library(tidyverse)

# import data
df <- read.csv("engagements-2021-08-11.csv")
df_rank <- read.csv("replays_info.csv") 
df_rank_old <- read.csv("replays_info_old.csv") %>% 
  select(ReplayID, GameLengthSeconds)

# wrangle data
df_rank <- df_rank %>% 
  inner_join(df_rank_old, by = c("ReplayID"))
df_rank_long <- df_rank %>% 
  mutate(uid_rank_1 = str_c(UID1, Race1, Rank1, sep = " "),
         uid_rank_2 = str_c(UID2, Race2, Rank2, sep = " ")) %>% 
  pivot_longer(cols = c("uid_rank_1", "uid_rank_2"),
               names_to = "uid_rank") %>% 
  mutate(win = as.numeric(str_detect(uid_rank, as.character(Winner)))) %>% 
  separate(value, sep = " ", into = c("UID", "Race", "Rank")) %>% 
  select(ReplayID, UID, Race, Rank, win, GameLengthSeconds)



df_rank_long <- df_rank_long %>% 
  separate(ReplayID, sep = "\\.", into = c("first", "second")) %>% 
  separate(first, sep = "_", into = c("pre", "num_id")) %>% 
  mutate(GameID = ifelse(str_detect(pre, "ggg"), 
                         str_c("ggg-", num_id),
                         str_c("st-", num_id))) %>% 
  select(-pre, -num_id, -second)

df_valid_rank <- df_rank_long %>% 
  filter(Rank != 0, Rank != 8)

write.csv(df_valid_rank, "replays_info_player.csv")

# join data
joined_df <- df %>% 
  left_join(df_valid_rank, c("GameID"))

joined_df <- joined_df %>% 
  mutate(League = fct_recode(Rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7")) %>% 
  filter(!is.na(League))
#color code
bronze_col <- "#4d4dff"
silver_col <- "#36b3b3"
gold_col <- "#884dff"
plat_col <- "#00e5e6"
diam_col <- "#3655b3"
mast_col <- "#000066"
gmast_col <- "#3c00b3"

# =================== League =================== #

scouting_num_replay <- joined_df %>% 
  group_by(League) %>% 
  summarise(num_replay = n_distinct(GameID) * 2)

bronze_num_replay <- (scouting_num_replay %>% filter(League == "Bronze"))[[1, 2]]
silver_num_replay <- (scouting_num_replay %>% filter(League == "Silver"))[[1, 2]]
gold_num_replay <- (scouting_num_replay %>% filter(League == "Gold"))[[1, 2]]
platinum_num_replay <- (scouting_num_replay %>% filter(League == "Platinum"))[[1, 2]]
diamond_num_replay <- (scouting_num_replay %>% filter(League == "Diamond"))[[1, 2]]
master_num_replay <- (scouting_num_replay %>% filter(League == "Master"))[[1, 2]]
gmaster_num_replay <- (scouting_num_replay %>% filter(League == "Grandmaster"))[[1, 2]]

# =================== When =================== #
bins <- 30
max_x <- 2500
max_y <- 3.5
y_title <- "Engagement Instances/Replay"
theme_set(theme_bw())
bronze <- ggplot(data = subset(joined_df, League == 'Bronze'), 
                 aes(x = StartTimeSeconds, y = ..count../bronze_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Bronze")  + 
  xlim(0, max_x) +
  ylim(0, max_y) +
  facet_wrap(~BaseClusterType)

silver <- ggplot(data = subset(joined_df, League == 'Silver'), 
                 aes(x = StartTimeSeconds, y = ..count../silver_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Silver") +
  xlim(0, max_x)+ 
  ylim(0, max_y)+
  facet_wrap(~BaseClusterType)

gold <- ggplot(data = subset(joined_df, League == 'Gold'), 
               aes(x = StartTimeSeconds, y = ..count../gold_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Gold") +
  xlim(0, max_x)+ 
  ylim(0, max_y)+
  facet_wrap(~BaseClusterType)

platinum <- ggplot(data = subset(joined_df, League == 'Platinum'), 
                   aes(x = StartTimeSeconds, y = ..count../platinum_num_replay)) +
  geom_histogram(color = plat_col, fill = plat_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Platinum") +
  xlim(0, max_x)+ 
  ylim(0, max_y)+
  facet_wrap(~BaseClusterType)

diamond <- ggplot(data = subset(joined_df, League == 'Diamond'), 
                  aes(x = StartTimeSeconds, y = ..count../diamond_num_replay)) +
  geom_histogram(color = diam_col, fill = diam_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Diamond") +
  xlim(0, max_x)+ 
  ylim(0, max_y)+
  facet_wrap(~BaseClusterType)

master <- ggplot(data = subset(joined_df, League == 'Master'), 
                 aes(x = StartTimeSeconds, y = ..count../master_num_replay)) +
  geom_histogram(color = mast_col, fill = mast_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Master") +
  xlim(0, max_x)+ 
  ylim(0, max_y)+
  facet_wrap(~BaseClusterType)

gmaster <- ggplot(data = subset(joined_df, League == 'Grandmaster'), 
                  aes(x = StartTimeSeconds, y = ..count../gmaster_num_replay)) +
  geom_histogram(color = gmast_col, fill = gmast_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Grandmaster") +
  xlim(0, max_x) + 
  ylim(0, max_y)+
  facet_wrap(~BaseClusterType)

grid.arrange(bronze, silver, gold, platinum, 
             diamond, master, gmaster)

# =================== Race =================== #

scouting_num_replay <- joined_df  %>% 
  group_by(Race) %>% 
  summarise(num_replay = n_distinct(GameID) * 2)

protoss_num_replay <- (scouting_num_replay %>% filter(Race == "Protoss"))[[1, 2]]  
terran_num_replay <- (scouting_num_replay %>% filter(Race == "Terran"))[[1, 2]]
zerg_num_replay <- (scouting_num_replay %>% filter(Race == "Zerg"))[[1, 2]]

bins <- 30
max_x <- 2500
max_y <- 3.5
y_title <- "Engagement Instances/Replay"


protoss <- ggplot(data = subset(joined_df, Race == 'Protoss'), 
                  aes(x = StartTimeSeconds, y = ..count../protoss_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Protoss") +
  xlim(0, max_x) + 
  ylim(0, max_y)

terran <- ggplot(data = subset(joined_df, Race == 'Terran'), 
                 aes(x = StartTimeSeconds, y = ..count../terran_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Terran") +
  xlim(0, max_x) + 
  ylim(0, max_y)

zerg <- ggplot(data = subset(joined_df, Race == 'Zerg'), 
               aes(x = StartTimeSeconds, y = ..count../zerg_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Zerg") +
  xlim(0, max_x) + 
  ylim(0, max_y)

grid.arrange(protoss, terran, zerg)

# =================== Army supply lost =================== #
bins <- 20
max_x <- 50
max_y <- 12
y_title <- "Engagement Instances/Replay"
x_title <- "Total supply unit lost"
bronze <- ggplot(data = subset(joined_df, League == 'Bronze'), 
                 aes(x = (ArmySupplyLost1 + ArmySupplyLost2 + 
                            WorkerSupplyLost1 + WorkerSupplyLost2), y = ..count../bronze_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Total supply unit lost-Bronze") +
  xlim(0, max_x) + 
  ylim(0, max_y) 
  
silver <- ggplot(data = subset(joined_df, League == 'Silver'), 
                   aes(x = (ArmySupplyLost1 + ArmySupplyLost2 + 
                              WorkerSupplyLost1 + WorkerSupplyLost2), y = ..count../silver_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Total supply unit lost-Silver") +
  xlim(0, max_x)+ 
  ylim(0, max_y)

gold <- ggplot(data = subset(joined_df, League == 'Gold'), 
               aes(x = (ArmySupplyLost1 + ArmySupplyLost2 + 
                          WorkerSupplyLost1 + WorkerSupplyLost2), y = ..count../gold_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Total supply unit lost-Gold") +
  xlim(0, max_x)+
  ylim(0, max_y)

platinum <- ggplot(data = subset(joined_df, League == 'Platinum'), 
                   aes(x = (ArmySupplyLost1 + ArmySupplyLost2 + 
                              WorkerSupplyLost1 + WorkerSupplyLost2), y = ..count../platinum_num_replay)) +
  geom_histogram(color = plat_col, fill = plat_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Total supply unit lost-Platinum") +
  xlim(0, max_x)+ 
  ylim(0, max_y)
diamond <- ggplot(data = subset(joined_df, League == 'Diamond'), 
                  aes(x = (ArmySupplyLost1 + ArmySupplyLost2 + 
                             WorkerSupplyLost1 + WorkerSupplyLost2), y = ..count../diamond_num_replay)) +
  geom_histogram(color = diam_col, fill = diam_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Total supply unit lost-Diamond") +
  xlim(0, max_x)+ 
  ylim(0, max_y)

master <- ggplot(data = subset(joined_df, League == 'Master'), 
                 aes(x = (ArmySupplyLost1 + ArmySupplyLost2 + 
                            WorkerSupplyLost1 + WorkerSupplyLost2), y = ..count../master_num_replay)) +
  geom_histogram(color = mast_col, fill = mast_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Total supply unit lost-Master") +
  xlim(0, max_x)+ 
  ylim(0, max_y)

gmaster <- ggplot(data = subset(joined_df, League == 'Grandmaster'), 
                  aes(x = (ArmySupplyLost1 + ArmySupplyLost2 + 
                             WorkerSupplyLost1 + WorkerSupplyLost2), y = ..count../gmaster_num_replay)) +
  geom_histogram(color = gmast_col, fill = gmast_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Total supply unit lost-Grandmaster") +
  xlim(0, max_x) + 
  ylim(0, max_y)

grid.arrange(bronze, silver, gold, platinum, 
             diamond, master, gmaster)


# =================== Base Type =================== #

scouting_num_replay <- joined_df  %>% 
  group_by(BaseClusterType) %>% 
  summarise(num_replay = n_distinct(GameID) * 2)

expansion_num_replay <- (scouting_num_replay %>% filter(BaseClusterType == "BaseType.EXPANSION"))[[1, 2]]  
main_num_replay <- (scouting_num_replay %>% filter(BaseClusterType == "BaseType.MAIN"))[[1, 2]]
proxy_num_replay <- (scouting_num_replay %>% filter(BaseClusterType == "BaseType.PROXY"))[[1, 2]]
none_num_replay <- (scouting_num_replay %>% filter(BaseClusterType == "BaseType.NONE"))[[1, 2]]

bins <- 30
max_x <- 2500
max_y <- 3.5
y_title <- "Engagement Instances/Replay"

expansion <- ggplot(data = subset(joined_df, BaseClusterType == "BaseType.EXPANSION"), 
                  aes(x = StartTimeSeconds, y = ..count../expansion_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Expansion") +
  xlim(0, max_x) + 
  ylim(0, max_y)

main <- ggplot(data = subset(joined_df, BaseClusterType == "BaseType.MAIN"), 
                 aes(x = StartTimeSeconds, y = ..count../main_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Main") +
  xlim(0, max_x) + 
  ylim(0, max_y)

proxy <- ggplot(data = subset(joined_df, BaseClusterType == "BaseType.PROXY"), 
               aes(x = StartTimeSeconds, y = ..count../proxy_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-Proxy") +
  xlim(0, max_x) + 
  ylim(0, max_y)

none <- ggplot(data = subset(joined_df, BaseClusterType == "BaseType.NONE"), 
                aes(x = StartTimeSeconds, y = ..count../none_num_replay)) +
  geom_histogram(color = plat_col, fill = plat_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Gametime-None") +
  xlim(0, max_x) + 
  ylim(0, max_y)

grid.arrange(expansion, main, proxy, none)

# =================== Duration =================== #
bins <- 30
max_x <- 50
max_y <- 2.5
y_title <- "Engagement Instances/Replay"

joined_df <- joined_df %>% 
  mutate(duration = EndTimeSeconds - StartTimeSeconds)

bronze <- ggplot(data = subset(joined_df, League == 'Bronze'), 
                 aes(x = duration, y = ..count../bronze_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Duration-Bronze")  + 
  xlim(0, max_x) +
  ylim(0, max_y) 

silver <- ggplot(data = subset(joined_df, League == 'Silver'), 
                 aes(x = duration, y = ..count../silver_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Duration-Silver") +
  xlim(0, max_x)+ 
  ylim(0, max_y)

gold <- ggplot(data = subset(joined_df, League == 'Gold'), 
               aes(x = duration, y = ..count../gold_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Duration-Gold") +
  xlim(0, max_x)+ 
  ylim(0, max_y)

platinum <- ggplot(data = subset(joined_df, League == 'Platinum'), 
                   aes(x = duration, y = ..count../platinum_num_replay)) +
  geom_histogram(color = plat_col, fill = plat_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Duration-Platinum") +
  xlim(0, max_x)+ 
  ylim(0, max_y)

diamond <- ggplot(data = subset(joined_df, League == 'Diamond'), 
                  aes(x = duration, y = ..count../diamond_num_replay)) +
  geom_histogram(color = diam_col, fill = diam_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Duration-Diamond") +
  xlim(0, max_x)+ 
  ylim(0, max_y)

master <- ggplot(data = subset(joined_df, League == 'Master'), 
                 aes(x = duration, y = ..count../master_num_replay)) +
  geom_histogram(color = mast_col, fill = mast_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Duration-Master") +
  xlim(0, max_x)+ 
  ylim(0, max_y)

gmaster <- ggplot(data = subset(joined_df, League == 'Grandmaster'), 
                  aes(x = duration, y = ..count../gmaster_num_replay)) +
  geom_histogram(color = gmast_col, fill = gmast_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Duration-Grandmaster") +
  xlim(0, max_x) + 
  ylim(0, max_y)

grid.arrange(bronze, silver, gold, platinum, 
             diamond, master, gmaster)

# =================== Workers lost =================== #
bins <- 30
max_x <- 40
max_y <- 3.8
y_title <- "Engagement Instances/Replay"

joined_df_filtered <- joined_df %>% 
  filter(WorkerSupplyLost1 + WorkerSupplyLost2 > 0)

bronze <- ggplot(data = subset(joined_df_filtered, League == 'Bronze'), 
                 aes(x = WorkerSupplyLost1 + WorkerSupplyLost2, y = ..count../bronze_num_replay)) +
  geom_histogram(color = bronze_col, fill = bronze_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Worker Supply Lost-Bronze") + 
  xlim(0, max_x) +
  ylim(0, max_y)

silver <- ggplot(data = subset(joined_df_filtered, League == 'Silver'), 
                 aes(x = WorkerSupplyLost1 + WorkerSupplyLost2, y = ..count../silver_num_replay)) +
  geom_histogram(color = silver_col, fill = silver_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Worker Supply Lost-Silver") +
  xlim(0, max_x)+
  ylim(0, max_y)

gold <- ggplot(data = subset(joined_df_filtered, League == 'Gold'), 
               aes(x = WorkerSupplyLost1 + WorkerSupplyLost2, y = ..count../gold_num_replay)) +
  geom_histogram(color = gold_col, fill = gold_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Worker Supply Lost-Gold") +
  xlim(0, max_x)+
  ylim(0, max_y)

platinum <- ggplot(data = subset(joined_df_filtered, League == 'Platinum'), 
                   aes(x = WorkerSupplyLost1 + WorkerSupplyLost2, y = ..count../platinum_num_replay)) +
  geom_histogram(color = plat_col, fill = plat_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Worker Supply Lost-Platinum") +
  xlim(0, max_x)+
  ylim(0, max_y)

diamond <- ggplot(data = subset(joined_df_filtered, League == 'Diamond'), 
                  aes(x = WorkerSupplyLost1 + WorkerSupplyLost2, y = ..count../diamond_num_replay)) +
  geom_histogram(color = diam_col, fill = diam_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Worker Supply Lost-Diamond") +
  xlim(0, max_x)+
  ylim(0, max_y)

master <- ggplot(data = subset(joined_df_filtered, League == 'Master'), 
                 aes(x = WorkerSupplyLost1 + WorkerSupplyLost2, y = ..count../master_num_replay)) +
  geom_histogram(color = mast_col, fill = mast_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Worker Supply Lost-Master") +
  xlim(0, max_x)+
  ylim(0, max_y)

gmaster <- ggplot(data = subset(joined_df_filtered, League == 'Grandmaster'), 
                  aes(x = WorkerSupplyLost1 + WorkerSupplyLost2, y = ..count../gmaster_num_replay)) +
  geom_histogram(color = gmast_col, fill = gmast_col,
                 bins = bins, alpha = 0.6) + 
  labs(y=y_title, x = "Worker Supply Lost-Grandmaster") + 
  xlim(0, max_x) +
  ylim(0, max_y)

grid.arrange(bronze, silver, gold, platinum, 
             diamond, master, gmaster, respect = TRUE)

# =================== Workers lost =================== #
num_bins <- 0.01
max_x <- 3800
max_y <- 0.02
y_title <- "Worker supply lost/Replay"

joined_df_filtered <- joined_df %>% 
  filter(WorkerSupplyLost1 + WorkerSupplyLost2 > 0)

bronze <- ggplot(data = subset(joined_df_filtered, League == 'Bronze'), 
                 aes(x = StartTimeSeconds, y = (WorkerSupplyLost1 + WorkerSupplyLost2)/bronze_num_replay)) +
  geom_col(color = bronze_col, fill = bronze_col,
           width = 100) + 
  labs(y=y_title, x = "Start Time (seconds)-Bronze") + 
  xlim(0, max_x) +
  ylim(0, max_y)

silver <- ggplot(data = subset(joined_df_filtered, League == 'Silver'), 
                 aes(x = StartTimeSeconds, y = (WorkerSupplyLost1 + WorkerSupplyLost2)/silver_num_replay)) +
  geom_col(color = silver_col, fill = silver_col,
           width = 100) + 
  labs(y=y_title, x = "Start Time (seconds)-Silver") + 
  xlim(0, max_x) +
  ylim(0, max_y)

gold <- ggplot(data = subset(joined_df_filtered, League == 'Gold'), 
               aes(x = StartTimeSeconds, y = (WorkerSupplyLost1 + WorkerSupplyLost2)/gold_num_replay)) +
  geom_col(color = gold_col, fill = gold_col,
           width = 100) + 
  labs(y=y_title, x = "Start Time (seconds)-Gold") + 
  xlim(0, max_x) +
  ylim(0, max_y)

platinum <- ggplot(data = subset(joined_df_filtered, League == 'Platinum'), 
                   aes(x = StartTimeSeconds, y = (WorkerSupplyLost1 + WorkerSupplyLost2)/platinum_num_replay)) +
  geom_col(color = plat_col, fill = plat_col,
           width = 100) + 
  labs(y=y_title, x = "Start Time (seconds)-Platinum") + 
  xlim(0, max_x) +
  ylim(0, max_y)

diamond <- ggplot(data = subset(joined_df_filtered, League == 'Diamond'), 
                  aes(x = StartTimeSeconds, y = (WorkerSupplyLost1 + WorkerSupplyLost2)/diamond_num_replay)) +
  geom_col(color = diam_col, fill = diam_col,
           width = 100) + 
  labs(y=y_title, x = "Start Time (seconds)-Diamond") + 
  xlim(0, max_x) +
  ylim(0, max_y)


master <- ggplot(data = subset(joined_df_filtered, League == 'Master'), 
                 aes(x = StartTimeSeconds, y = (WorkerSupplyLost1 + WorkerSupplyLost2)/master_num_replay)) +
  geom_col(color = mast_col, fill = mast_col,
           width = 100) + 
  labs(y=y_title, x = "Start Time (seconds)-Master") + 
  xlim(0, max_x) +
  ylim(0, max_y)

gmaster <- ggplot(data = subset(joined_df_filtered, League == 'Grandmaster'), 
                  aes(x = StartTimeSeconds, y = (WorkerSupplyLost1 + WorkerSupplyLost2)/gmaster_num_replay)) +
  geom_col(color = gmast_col, fill = gmast_col,
           width = 100) + 
  labs(y=y_title, x = "Start Time (seconds)-Grandmaster") + 
  xlim(0, max_x) +
  ylim(0, max_y)

grid.arrange(bronze, silver, gold, platinum, 
             diamond, master, gmaster, respect = TRUE)

# =================== Army value lost ~ win percentages =================== #

df_info <- df_rank %>% 
  separate(ReplayID, sep = "\\.", into = c("first", "second")) %>% 
  separate(first, sep = "_", into = c("pre", "num_id")) %>% 
  mutate(GameID = ifelse(str_detect(pre, "ggg"), 
                         str_c("ggg-", num_id),
                         str_c("st-", num_id))) %>% 
  select(-pre, -num_id, -second)

df_joined <- df %>% 
  left_join(df_info, c("GameID"))

df_joined_extended <- df_joined %>% 
  mutate(player_1 = str_c(UID1, Race1, Rank1, 
                          TotalArmyValue1, TotalArmySupply1,
                          ArmySupplyLost1, ArmyValueLost1,
                          TotalWorkerSupply1, WorkerSupplyLost1,
                          sep = " "),
         player_2 = str_c(UID2, Race2, Rank2, 
                          TotalArmyValue2, TotalArmySupply2,
                          ArmySupplyLost2, ArmyValueLost2,
                          TotalWorkerSupply2, WorkerSupplyLost2,
                          sep = " ")) %>% 
  pivot_longer(cols = c("player_1", "player_2"),
               names_to = "combined_field",
               values_to = "combined_value")  
  
df_joined_extended <- df_joined_extended %>%  
  mutate(win = as.numeric(str_detect(combined_field, as.character(Winner)))) %>% 
  separate(combined_value, sep = " ", into = c("UID", "Race", "Rank", 
                                      "ArmyValue", "ArmySupply",
                                      "ArmySupplyLost", "ArmyValueLost",
                                      "WorkerSupply", "WorkerSupplyLost")) 

df_joined_player <- df_joined_extended %>% 
  mutate(diff_army_value_lost_1 = ArmyValueLost1 - ArmyValueLost2,
         diff_army_value_lost_2 = ArmyValueLost2 - ArmyValueLost1,
         diff_army_value_lost = ifelse(str_detect(combined_field, "1"),
                                                  diff_army_value_lost_1,
                                                  diff_army_value_lost_2), 
         rel_total_army_value_1 = TotalArmyValue1 - TotalArmyValue2,
         rel_total_army_value_2 = TotalArmyValue2 - TotalArmyValue1,
         rel_total_army_value = ifelse(str_detect(combined_field, "1"),
                                       rel_total_army_value_1,
                                       rel_total_army_value_2),
         opponent_lost_worker = ifelse(str_detect(combined_field, "1"),
                                       WorkerSupplyLost2 > 0,
                                       WorkerSupplyLost1 > 0))
  


# df_joined_player_same_army_value <- df_joined_extended %>% 
#   mutate(diff_army_value_lost_1 = ArmyValueLost1 - ArmyValueLost2,
#          diff_army_value_lost_2 = ArmyValueLost2 - ArmyValueLost1,
#          diff_army_value_lost = ifelse(str_detect(combined_field, "1"),
#                                        diff_army_value_lost_1,
#                                        diff_army_value_lost_2), 
#          rel_total_army_value_1 = TotalArmyValue1 - TotalArmyValue2,
#          rel_total_army_value_2 = TotalArmyValue2 - TotalArmyValue1,
#          rel_total_army_value = ifelse(str_detect(combined_field, "1"),
#                                        rel_total_army_value_1,
#                                        rel_total_army_value_2),
#          opponent_lost_worker_1 = WorkerSupplyLost2 > 0, 
#          opponent_lost_worker_2 = WorkerSupplyLost1 > 0,
#          opponent_lost_worker = ifelse(str_detect(combined_field, "1"),
#                                        opponent_lost_worker_1,
#                                        opponent_lost_worker_2), 
#          Status = ifelse(BaseClusterPlayer == -1, "None",
#                          ifelse(str_detect(combined_field, 
#                                            as.character(BaseClusterPlayer)),
#                                 "Defense", "Offense")),
#          value_lost_1 = ifelse(Status == "Offense", 
#                              ArmyValueLost1, 
#                              ArmyValueLost1 + WorkerSupplyLost1 + BuildingCountLost1),
#          value_lost_2 = ifelse(Status == "Offense", 
#                                ArmyValueLost2, 
#                                ArmyValueLost2 + WorkerSupplyLost2 + BuildingCountLost2),
#          rel_value_lost = ifelse(str_detect(combined_field, "1"),
#                                  value_lost_1 - value_lost_2,
#                                  value_lost_2 - value_lost_1),
#          total_value_1 = ifelse(Status == "Offense", 
#                               TotalArmyValue1,
#                               TotalArmyValue1 + TotalWorkerSupply1 + TotalBuildingCount1),
#          total_value_2 = ifelse(Status == "Offense", 
#                                 TotalArmyValue2,
#                                 TotalArmyValue2+ TotalWorkerSupply2 + TotalBuildingCount2),
#          rel_total_value = ifelse(str_detect(combined_field, "1"),
#                                   total_value_1 - total_value_2,
#                                   total_value_2 - total_value_1)) %>% 
#   filter(abs(diff_army_value_lost) < 0.01*max(ArmyValueLost1, ArmyValueLost2))
# 
# 
# df_joined_player_same_army_value <- df_joined_player_same_army_value %>% 
#   filter(Rank != 0, Rank != 8) %>% 
#   mutate(League = fct_recode(Rank,
#                              "Bronze" = "1",
#                              "Silver" = "2",
#                              "Gold" = "3",
#                              "Platinum" = "4",
#                              "Diamond" = "5",
#                              "Master" = "6",
#                              "Grandmaster" = "7"))



df_joined_player <- df_joined_player %>% 
  filter(Rank != 0, Rank != 8) %>% 
  mutate(League = fct_recode(Rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7"))

df_joined_player_used_graph <- df_joined_player %>% 
  mutate(Status = ifelse(BaseClusterPlayer == -1, "None",
                         ifelse(str_detect(combined_field, 
                                           as.character(BaseClusterPlayer)),
                                "Defense", "Offense")),
         Status_worker_lost = ifelse(Status == "Offense", 
                                     ifelse(opponent_lost_worker, 
                                            "Offense_Enemy_Lose_Workers",
                                            "Offense_Enemy_Not_Lose_Workers"),
                                     Status),
         Status_army = ifelse(Status == "Offense", 
                              ifelse(rel_total_army_value > 0,
                                     "Offense_Greater_Army",
                                     "Offense_Smaller_Army"),
                              Status)) %>% 
  select(win, Status, Status_worker_lost, Status_army, League)
         
win_per <- df_joined_player %>% 
  group_by(League, diff_army_value_lost) %>% 
  summarise(win_per = mean(win))

num_bins <- 50
y_title <- "Win Percentage"

# win percentage ~ relative army value lost
ggplot(win_per, aes(x = diff_army_value_lost, y = win_per)) +
  geom_line(aes(color = League)) + 
  labs(y=y_title, x = "Relative army value lost") +
  facet_wrap(~League) +
  xlim(-10000, 10000)

win_per_base <- df_joined_player_used_graph %>% 
  group_by(League, Status) %>% 
  summarise(win_per = mean(win))

win_per_base_1 <- df_joined_player_used_graph %>% 
  group_by(League, Status_worker_lost) %>% 
  summarise(win_per = mean(win))

win_per_base_2 <- df_joined_player_used_graph %>% 
  group_by(League, Status_army) %>% 
  summarise(win_per = mean(win))

novice_col <- "#4d4dff"
prof_col <- "#36b3b3"
expert_col <- "#884dff"

bronze_col <- "#4d4dff"
silver_col <- "#36b3b3"
gold_col <- "#884dff"
plat_col <- "#00e5e6"

# win percentage ~ status
base <- ggplot(win_per_base, aes(x = Status, y = win_per)) +
  geom_col(aes(fill = Status)) + 
  geom_point() +
  geom_line(group = 1) +
  labs(y=y_title, x = "Status of engagement") +
  facet_wrap(~League) +
  coord_cartesian(ylim = c(0.3, 0.65)) +
  theme(axis.text.x=element_blank()) +
  scale_fill_manual(values = c(novice_col, prof_col, expert_col))

base_1 <- ggplot(win_per_base_1, aes(x = Status_worker_lost, y = win_per)) +
  geom_col(aes(fill = Status_worker_lost)) + 
  geom_point() +
  geom_line(group = 1) +
  labs(y=y_title, x = "Status of engagement") +
  facet_wrap(~League) +
  coord_cartesian(ylim = c(0.3, 1)) +
  theme(axis.text.x=element_blank()) +
  scale_fill_manual(values = c(bronze_col, silver_col, gold_col, plat_col))

base_2 <- ggplot(win_per_base_2, aes(x = Status_army, y = win_per)) +
  geom_col(aes(fill = Status_army)) + 
  geom_point() +
  geom_line(group = 1) +
  labs(y=y_title, x = "Status of engagement") +
  facet_wrap(~League) +
  coord_cartesian(ylim = c(0.3, 1)) +
  theme(axis.text.x=element_blank()) +
  scale_fill_manual(values = c(bronze_col, silver_col, gold_col, plat_col))

grid.arrange(base, base_1, base_2)

win_per_base_amry_lost <- df_joined_player %>% 
  group_by(League, Status, diff_army_value_lost) %>% 
  summarise(win_per = mean(win))

xmin <- -7500
xmax <- 7500

bronze <- ggplot(subset(win_per_base_amry_lost, League == 'Bronze'), 
                 aes(x = diff_army_value_lost, y = win_per)) +
  geom_line(color = bronze_col) + 
  labs(y=y_title, x = "Relative army value lost - Bronze") +
  facet_wrap(~Status) +
  xlim(xmin, xmax)

silver <- ggplot(subset(win_per_base_amry_lost, League == 'Silver'), 
                 aes(x = diff_army_value_lost, y = win_per)) +
  geom_line(color = silver_col) + 
  labs(y=y_title, x = "Relative army value lost - Silver") +
  facet_wrap(~Status) +
  xlim(xmin, xmax)

gold <- ggplot(subset(win_per_base_amry_lost, League == 'Gold'), 
                 aes(x = diff_army_value_lost, y = win_per)) +
  geom_line(color = gold_col) + 
  labs(y=y_title, x = "Relative army value lost - Gold") +
  facet_wrap(~Status) +
  xlim(xmin, xmax)

platinum <- ggplot(subset(win_per_base_amry_lost, League == 'Platinum'), 
                 aes(x = diff_army_value_lost, y = win_per)) +
  geom_line(color = plat_col) + 
  labs(y=y_title, x = "Relative army value lost - Platinum") +
  facet_wrap(~Status) +
  xlim(xmin, xmax)

diamond <- ggplot(subset(win_per_base_amry_lost, League == 'Diamond'), 
                 aes(x = diff_army_value_lost, y = win_per)) +
  geom_line(color = diam_col) + 
  labs(y=y_title, x = "Relative army value lost - Diamond") +
  facet_wrap(~Status) +
  xlim(xmin, xmax)

master <- ggplot(subset(win_per_base_amry_lost, League == 'Master'), 
                 aes(x = diff_army_value_lost, y = win_per)) +
  geom_line(color = mast_col) + 
  labs(y=y_title, x = "Relative army value lost - Master") +
  facet_wrap(~Status) +
  xlim(xmin, xmax)

gmaster <- ggplot(subset(win_per_base_amry_lost, League == 'Grandmaster'), 
                 aes(x = diff_army_value_lost, y = win_per)) +
  geom_line(color = gmast_col) + 
  labs(y=y_title, x = "Relative army value lost - Grandmaster") +
  facet_wrap(~Status) +
  xlim(xmin, xmax)

grid.arrange(bronze, silver, gold, platinum, 
             diamond, master, gmaster)

# total army value
total_rel_army_value <- df_joined_player_same_army_value %>% 
  group_by(League, Status) %>% 
  summarise(avg_rel_total_value = mean(rel_total_army_value))

ggplot(total_rel_army_value, aes(x = Status, y = avg_rel_total_value)) +
  geom_col(aes(fill = Status)) + 
  geom_point() +
  geom_line(group = 1) +
  labs(x = "Status of engagement") +
  facet_wrap(~League) 

# df_join_player_none <- df_joined_player %>% 
#   mutate(indicator = ifelse(BaseClusterType == "BaseType.NONE", 0, 1)) %>% 
#   group_by(GameID) %>% 
#   summarise(total = sum(indicator)) %>% 
#   filter(total == 0)


# attack
df_join_player_game <- df_joined_player %>% 
  count(GameID, UID, Status) %>% 
  filter(Status == "Offense")

names(df_join_player_game) <- c("game_id", "uid", "status", "offense_count")

all_stats_check <- all_stats_win %>% 
  inner_join(df_join_player_game, by = c("game_id", "uid")) 

%>% 
  filter(rank == 7)

ggplot(all_stats_check, aes(x = offense_count, y = scout_count)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm") +
  geom_jitter()

summary(lm(data = all_stats_check, offense_count ~ scout_count))

write.csv(df_joined_player, "engagement_data_player.csv")

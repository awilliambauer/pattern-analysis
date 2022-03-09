# library
library(dplyr)

# import data
df <- read.csv("scouting_stats_cluster.csv") %>% 
  filter(!is.na(Rank))

# wrangle data
df$Rank <- as.factor(df$Rank)
df <- df %>% 
  mutate(League = fct_recode(Rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7")) %>% 
  group_by(UID) %>% 
  filter(n_distinct(League) == 1) %>% 
  ungroup()

set.seed(525701)
bronze_UID <- (df %>% group_by(League) %>% 
                 filter(League == "Bronze") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 700))$UID
silver_UID <- (df %>% group_by(League) %>% 
                 filter(League == "Silver") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 700))$UID
gold_UID <- (df %>% group_by(League) %>% 
               filter(League == "Gold") %>% 
               distinct(UID) %>% 
               slice_sample(n = 700))$UID
platinum_UID <- (df %>% group_by(League) %>% 
                   filter(League == "Platinum") %>% 
                   distinct(UID) %>% 
                   slice_sample(n = 700))$UID
diamond_UID <- (df %>% group_by(League) %>% 
                  filter(League == "Diamond") %>% 
                  distinct(UID) %>% 
                  slice_sample(n = 700))$UID
master_UID <- (df %>% group_by(League) %>% 
                 filter(League == "Master") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 700))$UID
grandmaster_UID <- (df %>% group_by(League) %>% 
                      filter(League == "Grandmaster") %>% 
                      distinct(UID) %>% 
                      slice_sample(n = 700))$UID

uids <- unique(c(bronze_UID, silver_UID, gold_UID, platinum_UID, diamond_UID, 
                 master_UID, grandmaster_UID))

df_sample <- df %>% 
  filter(UID %in% uids)

# sanity check
df_test <- df_sample %>% 
  group_by(League) %>% 
  filter(League == "Platinum") %>% distinct(UID)

# ======================= # Warmup # ======================= #
ggplot(df, aes(x = League, y = Command_rate)) +
  geom_boxplot() +
  scale_y_log10()


# ======================= # Aggregate Group Control # ======================= #
ggplot(df_sample, aes(x = CPS, color = League, group = League)) +
  geom_histogram(aes(y=..density..), 
                 fill = NA, size = 1.5, 
                 position = 'identity', alpha = 0.5) +
  scale_color_hue(direction = -1)

# ggplot(df, aes(x = CPS, color = League, group = League)) +
#   geom_histogram(data = subset(df_sample, League == 'Bronze'),
#                  aes(y=..count../sum(..count..)), 
#                  fill = NA, size = 1.5, color = "pink") +
#   geom_histogram(data = subset(df_sample, League == 'Silver'),
#                  aes(y=..count../sum(..count..)), 
#                  fill = NA, size = 1.5, color = "purple") +
#   geom_histogram(data = subset(df_sample, League == 'Gold'),
#                  aes(y=..count../sum(..count..)), 
#                  fill = NA, size = 1.5, color = "blue") +
#   geom_histogram(data = subset(df_sample, League == 'Platinum'),
#                  aes(y=..count../sum(..count..)), 
#                  fill = NA, size = 1.5, color = "lightgreen") +
#   geom_histogram(data = subset(df_sample, League == 'Diamond'),
#                  aes(y=..count../sum(..count..)), 
#                  fill = NA, size = 1.5, color = "darkgreen") +
#   geom_histogram(data = subset(df_sample, League == 'Master'),
#                  aes(y=..count../sum(..count..)), 
#                  fill = NA, size = 1.5, color = "brown") +
#   geom_histogram(data = subset(df_sample, League == 'Grandmaster'),
#                  aes(y=..count../sum(..count..)), 
#                  fill = NA, size = 1.5, color = "red") +
#   scale_color_discrete(labels = c("Grandmaster", "Master", "Diamond", "Platinum",
#                                   "Gold", "Silver", "Bronze"))

bins <- 0.005
ggplot(df, aes(x = ScoutingFrequencyAfterFirstBattle, color = League, group = League)) +
  geom_histogram(data = subset(df, League == 'Bronze'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Silver'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Gold'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Platinum'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Diamond'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Master'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Grandmaster'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = bins) +
  xlim(0, 0.1)




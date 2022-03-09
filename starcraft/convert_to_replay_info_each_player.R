# library
library(tidyverse)

# import data
df <- read.csv("engagements-2021-08-11.csv")
df_rank <- read.csv("replays_info.csv") 
# add game length seconds
df_rank_old <- read.csv("replays_info_old.csv") %>% 
  select(ReplayID, GameLengthSeconds)

# wrangle data
df_rank <- df_rank %>% 
  inner_join(df_rank_old, by = c("ReplayID")) %>% 
  rename(GameLengthSeconds = GameLengthSeconds.y) %>% 
  select(-GameLengthSeconds.x)
df_rank_long <- df_rank %>% 
  mutate(uid_rank_1 = str_c(UID1, Race1, Rank1, Region1, sep = " "),
         uid_rank_2 = str_c(UID2, Race2, Rank2, Region2, sep = " ")) %>% 
  pivot_longer(cols = c("uid_rank_1", "uid_rank_2"),
               names_to = "uid_rank") %>% 
  mutate(win = as.numeric(str_detect(uid_rank, as.character(Winner)))) %>% 
  separate(value, sep = " ", into = c("UID", "Race", "Rank", "Region")) %>% 
  select(ReplayID, UID, Race, Rank, Region, win, GameLengthSeconds)



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

# Combine with all-stats data

all_stats_win_df <- read.csv("all_stats_win.csv")
all_stats_skill_df <- read.csv("all_stats_skill.csv")

df_region <- df_valid_rank %>% select("UID", "GameID", "Region")
df_region$UID <- as.integer(df_region$UID)

all_stats_win_df_combined <- all_stats_win_df %>% 
  inner_join(df_region, by = c("game_id" = "GameID", "uid" = "UID"))

all_stats_skill_df_combined <- all_stats_skill_df %>% 
  inner_join(df_region, by = c("game_id" = "GameID", "uid" = "UID"))

write.csv(all_stats_win_df_combined, "all_stats_win_combined.csv")
write.csv(all_stats_skill_df_combined, "all_stats_skill_combined.csv")

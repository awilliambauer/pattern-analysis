library(tidyverse)

common_stats <- read.csv("sc2_prediction_data-2021-08-08.csv") %>% 
  filter(!is.na(rank))
# macro_util <- read.csv("macro_utilization-2021-08-06.csv")
engagement_df <- read.csv("engagement_data_player.csv")
df_rank <- read.csv("replays_info_player.csv")
hotkey <- read.csv("camera_hotkeys-2021-08-09.csv")
hotkey_usage <- read.csv("camera_hotkeys_usage-2021-08-13.csv")
macro <- read.csv("macro_count-2021-08-10.csv")
scouting <- read.csv("scouting_instances_gm-2021-08-11.csv")

# win prediction data
hotkey_long <- hotkey %>% 
  mutate(uid_rank_1 = str_c(UID1, HotkeyCount1, sep = " "),
         uid_rank_2 = str_c(UID2, HotkeyCount2, sep = " ")) %>% 
  pivot_longer(cols = c("uid_rank_1", "uid_rank_2"),
               names_to = "uid_rank") %>% 
  separate(value, sep = " ", into = c("UID", "HotkeyCount")) %>% 
  mutate(RelHotkeyCount = ifelse(str_detect(uid_rank, "1"),
                                 HotkeyCount1 - HotkeyCount2,
                                 HotkeyCount2 - HotkeyCount1)) %>% 
  select(GameID, UID, HotkeyCount, RelHotkeyCount) 
names(hotkey_long) <- c("game_id", "uid", "hotkey_count", 
                            "rel_hotkey_count")

hotkey_usage_long <- hotkey_usage %>% 
  mutate(uid_rank_1 = str_c(UID1, HotkeyUsageCount1, sep = " "),
         uid_rank_2 = str_c(UID2, HotkeyUsageCount2, sep = " ")) %>% 
  pivot_longer(cols = c("uid_rank_1", "uid_rank_2"),
               names_to = "uid_rank") %>% 
  separate(value, sep = " ", into = c("UID", "HotkeyUsageCount")) %>% 
  mutate(RelHotkeyUsageCount = ifelse(str_detect(uid_rank, "1"),
                                      HotkeyUsageCount1 - HotkeyUsageCount2,
                                      HotkeyUsageCount2 - HotkeyUsageCount1)) %>% 
  select(GameID, UID, HotkeyUsageCount, RelHotkeyUsageCount) 
names(hotkey_usage_long) <- c("game_id", "uid", "hotkey_usage_count", 
                        "rel_hotkey_usage_count")


macro_long <- macro %>% 
  mutate(uid_rank_1 = str_c(UID1, MacroUtilization1, sep = " "),
         uid_rank_2 = str_c(UID2, MacroUtilization2, sep = " ")) %>% 
  pivot_longer(cols = c("uid_rank_1", "uid_rank_2"),
               names_to = "uid_rank") %>% 
  separate(value, sep = " ", into = c("UID", "MacroUtilization")) %>% 
  mutate(RelMacroUtilization = ifelse(str_detect(uid_rank, "1"),
                                 MacroUtilization1 - MacroUtilization2,
                                 MacroUtilization2 - MacroUtilization1)) %>% 
  select(GameID, UID, MacroUtilization, RelMacroUtilization) 
names(macro_long) <- c("game_id", "uid", "macro_count", 
                        "rel_macro_count")

scouting_long <- scouting %>% 
  group_by(GameID, UID) %>% 
  summarise(scout_count_all = n(),
            scout_count_engagement = sum(DuringEngagement == "False")) %>% 
  group_by(GameID) %>% 
  arrange(scout_count_all, .by_group = TRUE) %>% 
  mutate(temp_diff_scout_count_all = scout_count_all - lag(scout_count_all),
         temp_diff_scout_count_all_sum = sum(temp_diff_scout_count_all, 
                                             na.rm = TRUE),
         rel_scout_count_all = ifelse(is.na(temp_diff_scout_count_all),
                                      -temp_diff_scout_count_all_sum,
                                      temp_diff_scout_count_all_sum)) %>% 
  arrange(scout_count_engagement, .by_group = TRUE) %>% 
  mutate(temp_diff_scout_count_engagement = scout_count_engagement - 
           lag(scout_count_engagement),
         temp_diff_scout_count_engagement_sum = sum(temp_diff_scout_count_engagement, 
                                             na.rm = TRUE),
         rel_scout_count_engagement = ifelse(is.na(temp_diff_scout_count_engagement),
                                      -temp_diff_scout_count_engagement_sum,
                                      temp_diff_scout_count_engagement_sum))
scouting_long <- scouting_long %>% 
  select(GameID, UID, 
         scout_count_all, rel_scout_count_all, 
         scout_count_engagement, rel_scout_count_engagement)
names(scouting_long)[1:2] <- c("game_id", "uid")
names(scouting_long)[5:6] <- c("scout_count_no_engagement", 
                               "rel_scout_count_no_engagement")
scouting_long$uid <- as.character(scouting_long$uid)

common_stats$uid <- as.character(common_stats$uid)
common_stats$rank <- as.character(common_stats$rank)

df_rank <- df_rank %>% select(-X)
names(df_rank) <- c("uid", "race", "rank", "win", "game_length", "game_id")
df_rank <- df_rank %>% select(-rank, -win)
df_rank$uid <- as.character(df_rank$uid)


all_stats_win <- common_stats %>% 
  inner_join(df_rank, by = c("game_id", "uid")) %>% 
  inner_join(hotkey_long, by = c("game_id", "uid")) %>% 
  inner_join(hotkey_usage_long, by = c("game_id", "uid")) %>% 
  inner_join(macro_long, by = c("game_id", "uid")) %>% 
  inner_join(scouting_long, by = c("game_id", "uid")) %>% 
  filter(!is.na(rank))

all_stats_win <- all_stats_win %>% 
  mutate(hotkey_rate = as.numeric(hotkey_count)/game_length,
         rel_hotkey_rate = rel_hotkey_count/game_length,
         hotkey_usage_rate = as.numeric(hotkey_usage_count)/game_length,
         rel_hotkey_usage_rate = rel_hotkey_usage_count/game_length,
         macro_rate = as.numeric(macro_count)/game_length,
         rel_macro_rate = rel_macro_count/game_length,
         scout_rate_all = as.numeric(scout_count_all)/game_length,
         rel_scout_rate_all = rel_scout_count_all/game_length,
         scout_rate_engagement = as.numeric(scout_count_no_engagement)/game_length,
         rel_scout_rate_engagement = rel_scout_count_no_engagement/game_length)

write.csv(all_stats_win, "all_stats_win.csv")

# skill prediction data
engagement_df_filtered <- engagement_df %>% 
  filter(BaseClusterType != "BaseType.NONE")

engagement_df_filtered <- engagement_df_filtered %>% 
  mutate(abs_rel_total_army_value = abs(rel_total_army_value))

  

engagement_df_player <- engagement_df_filtered %>% 
  group_by(GameID, UID) %>% 
  summarise(abs_avg_rel_total_army_value = abs(mean(rel_total_army_value)),
            offense_lower_army_rate = sum(Status == "Offense" & 
                                             rel_total_army_value < 0)/sum(
                                               Status == "Offense"
                                             ))

names(engagement_df_player) <- c("game_id", "uid", "abs_rel_total_army",
                                 "offense_lower_army_rate")
engagement_df_player$uid <- as.character(engagement_df_player$uid)

all_stats_skill <- all_stats_win %>% 
  inner_join(engagement_df_player, by = c("game_id", "uid"))

all_stats_skill$offense_lower_army_rate[is.nan(all_stats_skill$offense_lower_army_rate)] <- 0

write.csv(all_stats_skill, "all_stats_skill.csv")
# write.csv(df_valid_rank, "replays_info_player.csv")

# total_rel_army_value <- engagement_df_filtered %>% 
#   group_by(Rank, Status) %>% 
#   summarise(avg_rel_total_value = abs(mean(rel_total_army_value)))
# 
# ggplot(engagement_df_filtered, aes(x = Status, y = rel_total_army_value)) +
#   geom_boxplot(aes(color = Status)) +
#   labs(x = "Status of engagement") +
#   facet_wrap(~ Rank) +
#   ylim(-5000, 5000)
# 
# check <- engagement_df_filtered %>% 
#   group_by(GameID, UID) %>% 
#   summarise(avg_abs_rel_total_army_value = mean(abs_rel_total_army_value))


# other data



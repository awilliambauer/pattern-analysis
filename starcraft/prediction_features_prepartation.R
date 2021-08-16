library(tidyverse)

common_stats <- read.csv("sc2_prediction_data-2021-08-08.csv") %>% 
  filter(!is.na(rank))
# macro_util <- read.csv("macro_utilization-2021-08-06.csv")
engagement_df <- read.csv("engagement_data_player.csv")
df_rank <- read.csv("replays_info_player.csv")
hotkey <- read.csv("camera_hotkeys-2021-08-09.csv")
macro <- read.csv("macro_count-2021-08-10.csv")

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

common_stats$uid <- as.character(common_stats$uid)
common_stats$rank <- as.character(common_stats$rank)

df_rank <- df_rank %>% select(-X)
names(df_rank) <- c("uid", "race", "rank", "win", "game_length", "game_id")
df_rank <- df_rank %>% select(-rank, -win)
df_rank$uid <- as.character(df_rank$uid)


all_stats_win <- common_stats %>% 
  inner_join(df_rank, by = c("game_id", "uid")) %>% 
  inner_join(hotkey_long, by = c("game_id", "uid")) %>% 
  inner_join(macro_long, by = c("game_id", "uid")) %>% 
  filter(!is.na(rank))

write.csv(all_stats_win, "all_stats_win.csv")

# skill prediction data
engagement_df_filtered <- engagement_df %>% 
  filter(BaseClusterType != "BaseType.NONE")

engagement_df_filtered <- engagement_df_filtered %>% 
  mutate(abs_rel_total_army_value = abs(rel_total_army_value))

engagement_df_player <- engagement_df_filtered %>% 
  group_by(GameID, UID) %>% 
  summarise(avg_abs_rel_total_army_value = abs(mean(rel_total_army_value)))

names(engagement_df_player) <- c("game_id", "uid", "rel_total_army")
engagement_df_player$uid <- as.character(engagement_df_player$uid)

all_stats_skill <- all_stats_win %>% 
  inner_join(engagement_df_player, by = c("game_id", "uid"))

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



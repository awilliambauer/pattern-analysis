# library
library(dplyr)

# import data
df <- read.csv("apm_stats_data2021-08-03.csv") %>% 
  filter(!is.na(rank))

# wrangle data
df$rank <- as.factor(df$rank)
df <- df %>% 
  mutate(League = fct_recode(rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7"))

# ======================= # Marco peace vs battle # ======================= #
# Save ratio 560 x 503
df_macro <- df %>% filter(apm_peace_macro/apm_battle_macro < 5)
ggplot(df_macro, aes(x = League, y = apm_peace_macro/apm_battle_macro)) +
  geom_boxplot(color = "#D40e2d") +
  theme_bw() +
  labs(y = "Peace time to Battle Macro Actions per Minute Ratio")

median_macro <- df %>% 
  group_by(League) %>% 
  summarise(Median_peace = median(apm_peace_macro),
            Median_battle = median(apm_battle_macro))

# ======================= # warmup vs non-warmup # ======================= #
df_warmup <- df %>% filter(apm_warmup/apm_non_warmup < 3)
ggplot(df_warmup, aes(x = League, y = apm_warmup/apm_non_warmup)) +
  geom_boxplot(color = "#D40e2d") +
  theme_bw() +
  labs(y = "Warmup to Non-warmup Actions per Minute Ratio")

# ======================= # Aggregate Group Control APM # ======================= #
order <- levels(df$League)
bins <- 30
size_border <- 1.75
ggplot(df, aes(x = apm, color = League, group = League)) +
  geom_histogram(data = subset(df, League == 'Bronze'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Silver'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Gold'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Platinum'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Diamond'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Master'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df, League == 'Grandmaster'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  xlim(0, 600) + 
  scale_color_manual(breaks = rev(order), 
                       values = rev(c("#ff00bf", "#5a00ff", "#008fff", "#01ff8c",
                                  "#5cff00", "#ffba03", "#ff0029"))) +
  theme_bw() + 
  labs(x = "Average actions per minute", 
       y = "Number of players (normalized)")
  

# ======= APM v Region ====== #

df_region <- read.csv("replays_info_player.csv") %>% 
  filter(Region != "")
df_region_rank <- df_region %>% count(Region, Rank)

ggplot(df_region_rank, aes(x = Rank, y = n)) + 
  geom_bar(stat = "identity") + 
  facet_wrap(~Region)

df_apm <- df %>% select(game_id, uid, apm, League)
df_region_rank_player <- df_region %>% 
  inner_join(df_apm, by = c("GameID" = "game_id", "UID" = "uid"))

# US 
df_region_rank_player_us <- df_region_rank_player %>% 
  filter(Region == "us")

us_graph <- ggplot(df_region_rank_player_us, aes(x = apm, color = League, group = League)) +
  geom_histogram(data = subset(df_region_rank_player_us, League == 'Bronze'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_us, League == 'Silver'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_us, League == 'Gold'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_us, League == 'Platinum'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_us, League == 'Diamond'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_us, League == 'Master'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_us, League == 'Grandmaster'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  xlim(0, 600) + 
  scale_color_manual(breaks = rev(order), 
                     values = rev(c("#ff00bf", "#5a00ff", "#008fff", "#01ff8c",
                                    "#5cff00", "#ffba03", "#ff0029"))) +
  theme_bw() + 
  labs(x = "Average actions per minute (US)", 
       y = "Number of players (normalized)")

# EU 
df_region_rank_player_eu <- df_region_rank_player %>% 
  filter(Region == "eu")

eu_graph <- ggplot(df_region_rank_player_eu, aes(x = apm, color = League, group = League)) +
  geom_histogram(data = subset(df_region_rank_player_eu, League == 'Bronze'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_eu, League == 'Silver'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_eu, League == 'Gold'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_us, League == 'Platinum'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_eu, League == 'Diamond'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_eu, League == 'Master'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  geom_histogram(data = subset(df_region_rank_player_eu, League == 'Grandmaster'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = size_border,
                 binwidth = bins) +
  xlim(0, 600) + 
  scale_color_manual(breaks = rev(order), 
                     values = rev(c("#ff00bf", "#5a00ff", "#008fff", "#01ff8c",
                                    "#5cff00", "#ffba03", "#ff0029"))) +
  theme_bw() + 
  labs(x = "Average actions per minute (EU)", 
       y = "Number of players (normalized)")

# combine 2 graphs
grid.arrange(us_graph, eu_graph)

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

# ======================= # Aggregate Group Control # ======================= #
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
  

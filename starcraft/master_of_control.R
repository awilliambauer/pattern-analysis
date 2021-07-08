# library
library(tidyverse)

# import data
moc <- read.csv("master_of_control_stats.csv") %>% 
  filter(!is.na(Rank))

# wrangle data
moc$Rank <- as.factor(moc$Rank)
moc <- moc %>% 
  mutate(League = fct_recode(Rank,
                             "Bronze" = "1",
                             "Silver" = "2",
                             "Gold" = "3",
                             "Platinum" = "4",
                             "Diamond" = "5",
                             "Master" = "6",
                             "Grandmaster" = "7"),
         Command_rate = CRWarmUp/CRNonWarmUp)

moc_sample <- moc %>% 
  group_by(League) %>% 
  slice_sample(n = 900) %>% 
  ungroup()

ggplot(moc_sample, aes(x = League, y = Command_rate)) +
  geom_boxplot() +
  scale_y_log10()
  
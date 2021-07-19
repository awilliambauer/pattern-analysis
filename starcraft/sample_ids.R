# Import libraries
library(tidyverse)
library(stringr)

# Import data
scouting <- read.csv("scouting_stats_apm.csv")

# Wrangle data
scouting <- scouting %>% 
  filter(Rank != "NaN") 

set.seed(525701)
scouting_sample <- scouting %>% 
  group_by(Rank) %>% 
  slice_sample(prop = 0.03)

sample_ids <- scouting_sample %>% 
  ungroup() %>% 
  select(Filename) %>% 
  distinct(Filename)

# Export to txt
write.table(sample_ids, file = "sample_ids.txt", sep = "\n", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)

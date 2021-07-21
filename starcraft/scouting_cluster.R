# library
library(tidyverse)

# import data
scouting <- read.csv("scouting_stats_cluster.csv") %>% 
  filter(!is.na(Rank))

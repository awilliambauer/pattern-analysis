# library
library(tidyverse)

# import data
moc <- read.csv("master_of_control_stats_new.csv") %>% 
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
         Command_rate = CRWarmUp/CRNonWarmUp) %>% 
  group_by(UID) %>% 
  filter(n_distinct(League) == 1) %>% 
  ungroup()

set.seed(525701)
bronze_UID <- (moc %>% group_by(League) %>% 
                 filter(League == "Bronze") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 50))$UID
silver_UID <- (moc %>% group_by(League) %>% 
                 filter(League == "Silver") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 50))$UID
gold_UID <- (moc %>% group_by(League) %>% 
               filter(League == "Gold") %>% 
               distinct(UID) %>% 
               slice_sample(n = 50))$UID
platinum_UID <- (moc %>% group_by(League) %>% 
                   filter(League == "Platinum") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 50))$UID
diamond_UID <- (moc %>% group_by(League) %>% 
                  filter(League == "Diamond") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 50))$UID
master_UID <- (moc %>% group_by(League) %>% 
                 filter(League == "Master") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 50))$UID
grandmaster_UID <- (moc %>% group_by(League) %>% 
                      filter(League == "Grandmaster") %>% 
                 distinct(UID) %>% 
                 slice_sample(n = 50))$UID

uids <- unique(c(bronze_UID, silver_UID, gold_UID, platinum_UID, diamond_UID, 
          master_UID, grandmaster_UID))

moc_sample <- moc %>% 
  filter(UID %in% uids)

# sanity check
moc_test <- moc_sample %>% 
  group_by(League) %>% 
  filter(League == "Platinum") %>% distinct(UID)

# ======================= # Warmup # ======================= #
ggplot(moc, aes(x = League, y = Command_rate)) +
  geom_boxplot() +
  scale_y_log10()
  

# ======================= # Aggregate Group Control # ======================= #
ggplot(moc_sample, aes(x = CPS, color = League, group = League)) +
  geom_histogram(aes(y=..density..), 
                 fill = NA, size = 1.5, 
                 position = 'identity', alpha = 0.5) +
  scale_color_hue(direction = -1)

ggplot(moc, aes(x = CPS, color = League, group = League)) +
  geom_histogram(data = subset(moc_sample, League == 'Bronze'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5, color = "pink") +
  geom_histogram(data = subset(moc_sample, League == 'Silver'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5, color = "purple") +
  geom_histogram(data = subset(moc_sample, League == 'Gold'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5, color = "blue") +
  geom_histogram(data = subset(moc_sample, League == 'Platinum'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5, color = "lightgreen") +
  geom_histogram(data = subset(moc_sample, League == 'Diamond'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5, color = "darkgreen") +
  geom_histogram(data = subset(moc_sample, League == 'Master'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5, color = "brown") +
  geom_histogram(data = subset(moc_sample, League == 'Grandmaster'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5, color = "red") +
  scale_color_discrete(labels = c("Grandmaster", "Master", "Diamond", "Platinum",
                                  "Gold", "Silver", "Bronze"))


ggplot(moc, aes(x = CPS, color = League, group = League)) +
  geom_histogram(data = subset(moc, League == 'Bronze'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = 0.4) +
  geom_histogram(data = subset(moc, League == 'Silver'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = 0.4) +
  geom_histogram(data = subset(moc, League == 'Gold'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = 0.4) +
  geom_histogram(data = subset(moc, League == 'Platinum'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = 0.4) +
  geom_histogram(data = subset(moc, League == 'Diamond'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = 0.4) +
  geom_histogram(data = subset(moc, League == 'Master'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = 0.4) +
  geom_histogram(data = subset(moc, League == 'Grandmaster'),
                 aes(y=..count../sum(..count..)), 
                 fill = NA, size = 1.5,
                 binwidth = 0.4) 


# ======================= # War vs Peace, Macro vs Micro # ======================= #
temp_moc <- moc %>% filter(PeaceRate/BattleRate < 15, PeaceRate < 10, 
                           BattleRate < 10)
ggplot(moc, aes(x = League, y = PeaceRate/BattleRate)) +
  geom_boxplot() +
  scale_y_log10()

ggplot(temp_moc, aes(x = League, y = PeaceRate)) +
  geom_boxplot() +
  scale_y_log10()

ggplot(temp_moc, aes(x = League, y = BattleRate)) +
  geom_boxplot() +
  scale_y_log10()

median_macro <- moc %>% 
  group_by(League) %>% 
  summarise(Median_peace = median(PeaceRate),
            Median_battle = median(BattleRate))

# ======= debuging ======= #

moc_gm <- moc %>% filter(League == "Grandmaster", 
                         PeaceRate == 0 | BattleRate == 0) %>% 
  select(Filename)
# Export to txt
write.table(moc_gm, file = "debug_ids.txt", sep = "\n", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)

moc_gm_csv <- read_csv("master_of_control_stats_debug.csv")
cg_check_1 <- (moc_gm_csv %>% filter(Filename == "spawningtool_47754.SC2Replay") %>% select(RelRank))$RelRank[1]

# ======================= # Warmup # ======================= #
ggplot(moc, aes(x = League, y = PeaceRb/BattleRb)) +
  geom_boxplot() + 
  scale_y_log10()

median_rebind <- moc %>% 
  group_by(League) %>% 
  summarise(Median_peace = median(PeaceRb),
            Median_battle = median(BattleRb))

# ======================= # Skill classification # ======================= #
# test <- "[213, 2321, 231312]"
# arr <- as.numeric(unlist(str_split(str_sub(test, 2, str_length(test)-1), ", ")))


toList <- function(x) {
  result <- vector(mode = "list", length = length(x))
  for(i in 1:length(x)) {
    arr <- as.numeric(unlist(str_split(str_sub(x[i], 2, str_length(x[i])-1), ", ")))
    result[[i]] <- arr
  }
  return(result)
}

# check <- toList(moc$CommandPerSec)
euclidean <- function(a, b) sqrt(sum((a - b)^2))


diffAll <- function(x) {
  res <- vector()
  for(i in 1:(length(x)-1)) {
    for(j in (i+1):length(x)) {
      euDiff <- euclidean(x[[i]], x[[j]])
      res <- append(res, euDiff)
    }
  }
  return(res)
}

res <- diffAll(toList(moc$CommandPerSec))

diff_moc_sample <- moc_sample %>% 
  group_by(League) %>% 
  summarise(Mean = mean(diffAll(toList(CommandPerSec))),
            Median = median(diffAll(toList(CommandPerSec))),
            SD = sd(diffAll(toList(CommandPerSec))),
            Min = min(diffAll(toList(CommandPerSec))),
            Max = max(diffAll(toList(CommandPerSec))))


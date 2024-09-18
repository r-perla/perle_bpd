rm(list = objects())
library(tidyverse)
library(lmerTest)
library(readr)

# Load data
d <- read_rds("Replication/Data/lisa/lisa_data.rds")

# Try this with Koens data
#d <- read_rds("Replication/Data/experiments/experiment_one.rds")

# Turn NaN into NA
d$expectation[is.nan(d$expectation)] <- NA
d$mean_population[is.nan(d$mean_population)] <- NA

# Ensure that profiles go from 1 to the number of profiles
d$profile <- d$profile - min(d$profile) + 1

# Omit NAs
d <- na.omit(d)

# Convert all variables to numeric
d <- d %>%
  mutate_all(as.numeric)

# Order data by participant, profile, item number, trial in this order
d <- d %>%
  ungroup() %>%
  select(participant, profile, factor_number, trial, expectation, feedback) %>%
  arrange(participant, profile, factor_number, trial)

# Lag expectation and feedback
d <- d %>%
  group_by(participant, profile, factor_number) %>%
  mutate(lag_expectation = lag(expectation),
         lag_feedback = lag(feedback)) %>%
  ungroup() %>%
  na.omit()

# Show head
head(d, 20)

# Predict expectation by lagged feedback
fit <- lmer(expectation ~ lag_feedback + (1 | participant), data = d)
summary(fit)

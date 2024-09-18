rm(list = objects())
library(tidyverse)
library(jtools)
library(reshape2)
library(rstan)
library(readr)

# Load data and model
model <- read_rds("Replication/fitted_models/fine_grain_better_fit.rds")
d <- read_rds("Replication/Data/experiments/experiment_one.rds") %>%
  na.omit()

# Extract predictions from model
preds <- colMeans(extract(model, pars = "y_rep")$y_rep)

# Add to data
d$expectation_pred <- preds

# Calculate absolute error
d$expectation_abs_error <- abs(d$expectation - d$expectation_pred)
d$residual <- d$expectation - d$expectation_pred

# Plot histogram of residuals
d %>%
  ggplot(aes(x = residual)) +
  geom_histogram(bins = 30, color = "black", fill = "lightblue") +
  labs(x = "Residual", y = "Frequency") +
  theme_apa()

# Do some EDA to investigate the cases with the highest error
high_error_d <- d %>%
  arrange(desc(expectation_abs_error)) %>%
  head(10) %>%
  select(participant, trial, expectation, expectation_pred, expectation_abs_error, residual, everything())

# Summarize error by participant
error_summary <- d %>%
  group_by(participant) %>%
  summarize(mean_error = mean(expectation_abs_error)) %>%
  arrange(desc(mean_error))

# Reduce data to worst participant for testing
worst_participant <- error_summary$participant[50]
d_worst <- d %>%
  filter(participant == worst_participant)

# Plot expectation by population mean
d_worst %>%
  ggplot(aes(x = expectation, y = mean_population)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Expectation", y = "Population mean") +
  theme_apa()

# Residuals
d_worst %>%
  ggplot(aes(x = residual)) +
  geom_histogram(bins = 30, color = "black", fill = "lightblue") +
  labs(x = "Residuals", y = "Frequency") +
  theme_apa()

# Expectations
d_worst %>%
  ggplot(aes(x = expectation)) +
  geom_histogram(bins = 30, color = "black", fill = "lightblue") +
  labs(x = "Expectation", y = "Frequency") +
  theme_apa()

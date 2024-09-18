rm(list = objects())
gc()
library(tidyverse)
library(jtools)
library(reshape2)
library(R.matlab)
library(rstan)
library(bayesplot)
library(readr)
set.seed(NULL)

# Load koens's data
raw_koen_alphas <- readMat("m4.mat")$m4

# For each list in koen_alphas, take the first value
koen_alphas <- c()

for (i in seq_along(raw_koen_alphas)) {
  koen_alphas <- c(koen_alphas, raw_koen_alphas[[i]][[1]][1])
}

raw_d <- read_rds("Replication/Data/experiments/experiment_one.rds") %>%
  na.omit()

# Load my fitted model
model <- read_rds("Replication/fitted_models/fine_grain_fit.rds")

# Extract alphas
alphas <- colMeans(extract(model, pars = "alpha")$alpha)

# Get ids
ids <- unique(raw_d$participant)

# Bind together with alphas
results <- data.frame(
  id = ids,
  my_alphas = alphas,
  koen_alphas = koen_alphas
)

# Mets
results_melted <- melt(results, id.vars = "id")

# Plot alphas (scatterplot)
results %>%
  ggplot(aes(x = my_alphas, y = koen_alphas)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "Stan alphas", y = "Optimization alphas") +
  theme_apa()

# Plot alphas density
results_melted %>%
  ggplot(aes(x = value, fill = variable)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("red", "blue"), labels = c("Stan", "Optimization")) +
  labs(x = "Alphas", y = "Density") +
  theme_apa()

raw_d %>%
  ggplot(aes(x = expectation, fill = as.factor(factor_number))) +
  geom_density(alpha = 0.5) +
  facet_wrap(~as.factor(factor_number)) +
  labs(x = "Expectation", y = "Density")

raw_d %>%
  ggplot(aes(x = expectation, fill = as.factor(factor_number))) +
  geom_histogram(alpha = 0.5) +
  facet_wrap(~as.factor(item_number), scales = "free_y") +
  labs(x = NULL, y = NULL) +
  theme_minimal()
  
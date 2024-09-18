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

# Load Lisa's data
#lisa_alphas <- c(unname(unlist(readMat("Replication/lisa_data/fineGran.mat"))))
lisa_alphas <- unname(readMat("Replication/lisa_data/fineGran_0.mat"))[[1]][, 1]
lise_exps <- unname(readMat("Replication/lisa_data/fineGran_0.mat"))[[1]][, 2]
raw_d <- read_rds("Replication/Data/lisa/lisa_data.rds")

# Load my fitted model
model <- read_rds("Replication/fitted_models_lisa/fine_grain_fit.rds")

# Extract alphas and exps
alphas <- colMeans(extract(model, pars = "alpha")$alpha)
exps <- colMeans(extract(model, pars = "start_expectation")$start_expectation)

# Get ids
ids <- unique(raw_d$participant)

# Get R optim alphas
optim_data <- read_rds("Replication/fitted_models_lisa/optim_param_estimates.rds")
optim_alphas <- optim_data$alpha
optim_exps <- optim_data$initial_exp

# Bind together with alphas
results <- data.frame(
  id = ids,
  my_alphas = alphas,
  lisa_alphas = lisa_alphas,
  r_optim_alphas = optim_alphas
)

# Melt
results_melted <- melt(results, id.vars = "id")

exp_results <- data.frame(
  id = ids,
  my_exps = exps,
  lisa_exps = lise_exps,
  r_optim_exps = optim_exps
)

# Melt
exp_results_melted <- melt(exp_results, id.vars = "id")

# Plot alphas (scatterplot)
results %>%
  ggplot(aes(x = my_alphas, y = lisa_alphas)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "Stan alphas", y = "Optimization alphas") +
  theme_apa()

results %>%
  ggplot(aes(x = lisa_alphas, y = r_optim_alphas)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "Stan alphas", y = "R optim alphas") +
  theme_apa()

# Plot alphas density
results_melted %>%
  ggplot(aes(x = value, fill = variable)) +
  geom_density(alpha = 0.5, color ="black") +
  scale_fill_manual(values = c("red", "blue", "yellow"), 
                    labels = c("Stan", "Matlab Optimization", "R Optimization")) +
  labs(x = "Alphas", y = "Density") +
  theme_apa()


# Plot exps (scatterplot)
exp_results %>%
  ggplot(aes(x = my_exps, y = lisa_exps)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "Stan exps", y = "Matlab exps") +
  theme_apa()

exp_results %>%
  ggplot(aes(x = lisa_exps, y = r_optim_exps)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "Stan exps", y = "R optim exps") +
  theme_apa()

# Plot exps density
exp_results_melted %>%
  ggplot(aes(x = value, fill = variable)) +
  geom_density(alpha = 0.5, color ="black") +
  scale_fill_manual(values = c("red", "blue", "yellow"), 
                    labels = c("Stan", "Matlab Optimization", "R Optimization")) +
  labs(x = "Exps", y = "Density") +
  theme_apa()

# Density as facet plot
exp_results_melted %>%
  ggplot(aes(x = value, fill = variable)) +
  geom_density(alpha = 0.5, color ="black") +
  scale_fill_manual(values = c("red", "blue", "yellow"), 
                    labels = c("Stan", "Matlab Optimization", "R Optimization")) +
  labs(x = "Exps", y = "Density") +
  facet_wrap(~variable) +
  theme_apa()

# As a historgram with method as facet
exp_results_melted %>%
  ggplot(aes(x = value, fill = variable)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 7, color = "black") +
  scale_fill_manual(values = c("red", "blue", "yellow"), 
                    labels = c("Stan", "Matlab Optimization", "R Optimization")) +
  labs(x = "Exps", y = "Density") +
  facet_wrap(~variable) +
  theme_apa()

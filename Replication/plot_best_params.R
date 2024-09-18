rm(list = objects())
gc()
library(tidyverse)
library(reshape2)
library(jtools)
library(rstan)
library(emmeans)
library(readr)

# Load data and best model files
d <- read_rds("Replication/Data/lisa/lisa_data.rds") %>%
  na.omit()
hc_hc_ft <- read_rds("Replication/fitted_models_lisa/hc_on_hc/fine_grain_better_fit.rds")
hc_bpd_fit <- read_rds("Replication/fitted_models_lisa/hc_on_bpd/fine_grain_better_fit.rds")
bpd_hc_fit <- read_rds("Replication/fitted_models_lisa/bpd_on_hc/fine_grain_better_fit.rds")
bpd_bpd_fit <- read_rds("Replication/fitted_models_lisa/bpd_on_bpd/fine_grain_better_fit.rds")

# Extract parameters
hc_hc_alpha <- colMeans(extract(hc_hc_ft, pars = "alpha")$alpha)
hc_hc_gamma <- colMeans(extract(hc_hc_ft, pars = "gamma")$gamma)
hc_hc_start_exp <- colMeans(extract(hc_hc_ft, pars = "start_expectation")$start_expectation)
hc_hc_sigma <- colMeans(extract(hc_hc_ft, pars = "sigma")$sigma)

hc_bpd_alpha <- colMeans(extract(hc_bpd_fit, pars = "alpha")$alpha)
hc_bpd_gamma <- colMeans(extract(hc_bpd_fit, pars = "gamma")$gamma)
hc_bpd_start_exp <- colMeans(extract(hc_bpd_fit, pars = "start_expectation")$start_expectation)
hc_bpd_sigma <- colMeans(extract(hc_bpd_fit, pars = "sigma")$sigma)

bpd_hc_alpha <- colMeans(extract(bpd_hc_fit, pars = "alpha")$alpha)
bpd_hc_gamma <- colMeans(extract(bpd_hc_fit, pars = "gamma")$gamma)
bpd_hc_start_exp <- colMeans(extract(bpd_hc_fit, pars = "start_expectation")$start_expectation)
bpd_hc_sigma <- colMeans(extract(bpd_hc_fit, pars = "sigma")$sigma)

bpd_bpd_alpha <- colMeans(extract(bpd_bpd_fit, pars = "alpha")$alpha)
bpd_bpd_gamma <- colMeans(extract(bpd_bpd_fit, pars = "gamma")$gamma)
bpd_bpd_start_exp <- colMeans(extract(bpd_bpd_fit, pars = "start_expectation")$start_expectation)
bpd_bpd_sigma <- colMeans(extract(bpd_bpd_fit, pars = "sigma")$sigma)

# Combine params to dataframes
hc_hc_df <- data.frame(
  alpha = hc_hc_alpha,
  gamma = hc_hc_gamma,
  sigma = hc_hc_sigma,
  data = "hc_hc"
)

hc_bpd_df <- data.frame(
  alpha = hc_bpd_alpha,
  gamma = hc_bpd_gamma,
  sigma = hc_bpd_sigma,
  data = "hc_bpd"
)

bpd_hc_df <- data.frame(
  alpha = bpd_hc_alpha,
  gamma = bpd_hc_gamma,
  sigma = bpd_hc_sigma,
  data = "bpd_hc"
)

bpd_bpd_df <- data.frame(
  alpha = bpd_bpd_alpha,
  gamma = bpd_bpd_gamma,
  sigma = bpd_bpd_sigma,
  data = "bpd_bpd"
)

# Seperate start exp dataframes
hc_hc_exps <- data.frame(
  start_exp = hc_hc_start_exp,
  data = "hc_hc"
)

hc_bpd_exps <- data.frame(
  start_exp = hc_bpd_start_exp,
  data = "hc_bpd"
)

bpd_hc_exps <- data.frame(
  start_exp = bpd_hc_start_exp,
  data = "bpd_hc"
)

bpd_bpd_exps <- data.frame(
  start_exp = bpd_bpd_start_exp,
  data = "bpd_bpd"
)

# Merge
all_params <- bind_rows(hc_hc_df, hc_bpd_df, bpd_hc_df, bpd_bpd_df)
all_exps <- bind_rows(hc_hc_exps, hc_bpd_exps, bpd_hc_exps, bpd_bpd_exps)

# Save to file
write_rds(all_params, "Replication/best_params_lisa_fine_rp.rds", compress = "gz")
write_rds(all_exps, "Replication/best_exps_lisa_fine_rp.rds", compress = "gz")

# Load in data
all_params <- read_rds("Replication/best_params_lisa_fine_rp.rds")
all_exps <- read_rds("Replication/best_exps_lisa_fine_rp.rds")

# Plot alpha density by source
all_params %>%
  ggplot(aes(x = alpha, fill = data)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density plot of alpha by source", x = "Alpha", y = "Density") +
  theme_apa()

# Same for gamma
all_params %>%
  ggplot(aes(x = gamma, fill = data)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density plot of gamma by source", x = "Gamma", y = "Density") +
  theme_apa()

# Same for start_exp
all_exps %>%
  ggplot(aes(x = start_exp, fill = data)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density plot of V0 by source", x = "Starting Expectation", y = "Density") +
  theme_apa()

# Same for sigma
all_params %>%
  ggplot(aes(x = sigma, fill = data)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density plot of sigma by source", x = "Sigma", y = "Density") +
  theme_apa()

# Run 2x2 ANOVA on alpha
aov_alpha <- aov(alpha ~ data, data = all_params)
summary(aov_alpha)
emmeans(aov_alpha, pairwise ~ data)

# Run 2x2 ANOVA on gamma
aov_gamma <- aov(gamma ~ data, data = all_params)
summary(aov_gamma)
emmeans(aov_gamma, pairwise ~ data)

# Run 2x2 ANOVA on start_exp
aov_start_exp <- aov(start_exp ~ data, data = all_exps)
summary(aov_start_exp)
emmeans(aov_start_exp, pairwise ~ data)

# Run 2x2 ANOVA on sigma
aov_sigma <- aov(sigma ~ data, data = all_params)
summary(aov_sigma)
emmeans(aov_sigma, pairwise ~ data)

# Same as violin plots
all_params %>%
  mutate(data = factor(data, levels = c("bpd_bpd", "bpd_hc", "hc_bpd", "hc_hc"),
                       labels = c("BPD-BPD", "BPD-HC", "HC-BPD", "HC-HC"))) %>%
  ggplot(aes(x = data, y = alpha, fill = data)) +
  geom_violin(trim = FALSE, alpha = 0.8, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Distribution of Alpha by Source",
       x = "Source", 
       y = "Alpha Value") +
  jtools::theme_apa() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

all_params %>%
  mutate(data = factor(data, levels = c("bpd_bpd", "bpd_hc", "hc_bpd", "hc_hc"),
                       labels = c("BPD-BPD", "BPD-HC", "HC-BPD", "HC-HC"))) %>%
  ggplot(aes(x = data, y = gamma, fill = data)) +
  geom_violin(trim = FALSE, alpha = 0.8, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Distribution of Gamma by Source",
       x = "Source", 
       y = "Gamma Value") +
  jtools::theme_apa() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

all_exps %>%
  mutate(data = factor(data, levels = c("bpd_bpd", "bpd_hc", "hc_bpd", "hc_hc"),
                       labels = c("BPD-BPD", "BPD-HC", "HC-BPD", "HC-HC"))) %>%
  ggplot(aes(x = data, y = start_exp, fill = data)) +
  geom_violin(trim = FALSE, alpha = 0.8, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Distribution of Starting Expectation by Source",
       x = "Source", 
       y = "Starting Expectation") +
  jtools::theme_apa() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

all_params %>%
  mutate(data = factor(data, levels = c("bpd_bpd", "bpd_hc", "hc_bpd", "hc_hc"),
                       labels = c("BPD-BPD", "BPD-HC", "HC-BPD", "HC-HC"))) %>%
  ggplot(aes(x = data, y = sigma, fill = data)) +
  geom_violin(trim = FALSE, alpha = 0.8, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Distribution of Sigma by Source",
       x = "Source", 
       y = "Sigma Value") +
  jtools::theme_apa() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

item_ref <- d %>%
  select(item_number, mean_population) %>%
  group_by(item_number) %>%
  summarize_all(mean)

plot(item_ref$mean_population, subset(all_exps, data == "hc_hc")$start_exp)

data.frame(ref = item_ref$mean_population, exps = subset(all_exps, data == "hc_hc")$start_exp) %>%
  ggplot(aes(x = ref, y = exps)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Starting Expectation by Item Reference",
       x = "Item Reference", 
       y = "Starting Expectation") +
  jtools::theme_apa()

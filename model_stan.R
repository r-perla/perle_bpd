rm(list = objects())
gc()
library(tidyverse)
library(lmerTest)
library(rstan)
library(readr)
set.seed(NULL)
rstan_options(auto_write = TRUE)

# Set path to save fitted models
out_path <- "Replication/fitted_models"

# Create path if it does not exist
if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

# Load data
d <- read_rds("Replication/Data/experiments/experiment_one.rds")
sim_mat <- read_rds("Replication/Data/experiments/similarity_matrix.rds")

# Turn NaN into NA
d$expectation[is.nan(d$expectation)] <- NA
d$mean_population[is.nan(d$mean_population)] <- NA

# Compile models
no_learning <- stan_model("Replication/Stan/no_learning.stan")
course_gran <- stan_model("Replication/Stan/course_gran.stan")
course_rp <- stan_model("Replication/Stan/course_rp.stan")
fine_grain <- stan_model("Replication/Stan/fine_grain.stan")
fine_grain_better <- stan_model("Replication/Stan/fine_grain_better.stan")
fine_rp <- stan_model("Replication/Stan/fine_rp.stan")

# Ensure that profiles go from 1 to the number of profiles
d$profile <- d$profile - min(d$profile) + 1

# Prepare data for Stan
P <- length(unique(d$participant))
N <- length(unique(d$trial))
Pr <- length(unique(d$profile))
n_factors <- length(unique(d$factor))

# Create data for RL models
rl_d <- na.omit(d)

# Calculate populations means across factors
pop_means <- rl_d %>%
  select(factor_number, mean_population) %>%
  group_by(factor_number) %>%
  summarize_all(mean)

# Extract variables for stan
R <- nrow(rl_d)
rl_est <- rl_d$expectation
rl_ref <- pop_means$mean_population
rl_feed <- rl_d$feedback
participant <- rl_d$participant
profile_id <- rl_d$profile
factor <- as.integer(rl_d$factor_number)
rl_reference <- rl_d$mean_population

# Create rl stan data
rl_standata <- list(R = R, N = N, P = P, Pr = Pr, n_F = n_factors, est = rl_est,
                    feedback = rl_feed, participant = participant, profile_id = profile_id, factor = factor)

# Create course_rp data
course_rp_data <- list(R = R, N = N, P = P, Pr = Pr, n_F = n_factors, est = rl_est,
                       feedback = rl_feed, participant = participant, profile_id = profile_id, factor = factor,
                       reference = rl_reference)

# Create fine_grain data
item_data <- rl_d %>%
  select(item_number, mean_population) %>%
  arrange(item_number) %>%
  unique()
item_data$item_number <- as.integer(item_data$item_number)

n_I <- length(item_data$item_number)
fine_start <- item_data$mean_population

# Subset similarity matrix to only the items numbers actually in the data
sim_mat <- sim_mat[item_data$item_number, item_data$item_number]

# Rename item_number
old_names <- sort(item_data$item_number)
new_names <- 1:n_I

# Create a named vector for the mapping
item_number_map <- setNames(new_names, old_names)

# Rename in rl_d
rl_d$item_number <- as.integer(item_number_map[as.character(rl_d$item_number)])
item <- rl_d$item_number

fine_data <- list(R = R, N = N, P = P, Pr = Pr, n_I = n_I, est = rl_est,
                  feedback = rl_feed, participant = participant, profile_id = profile_id, 
                  item = item, sim_mat = sim_mat)

fine_rp_data <- list(R = R, N = N, P = P, Pr = Pr, n_I = n_I, est = rl_est,
                     feedback = rl_feed, participant = participant, profile_id = profile_id, 
                     item = item, sim_mat = sim_mat, reference = rl_reference)


# Create data list
no_learning_data <- list(R = R, P = P, est = rl_est, reference = rl_reference, participant = participant)

# Run no learning stan model
no_learning_fit <- sampling(no_learning, data = no_learning_data, chains = 8, cores = 8,
                            iter = 3000, warmup = 1500, pars = c("b0", "b1", "sigma", "y_rep", "log_lik"),
                            include = TRUE)

# Save model
write_rds(no_learning_fit, file.path(out_path, "no_learning_fit.rds"), compress = "gz")

# Remove model from memory
rm(no_learning_fit)
gc()

# Fit course granularity model
course_gran_fit <- sampling(course_gran, data = rl_standata, chains = 8, cores = 8,
                            iter = 2000, warmup = 1000, pars = c("alpha", "sigma", 
                                                                 "start_expectation", "log_lik", "y_rep"),
                            include = TRUE)

# Save model
write_rds(course_gran_fit, file.path(out_path, "course_gran_fit.rds"), compress = "gz")

# Remove model from memory
rm(course_gran_fit)
gc()

# Fir course RP model
course_rp_fit <- sampling(course_rp, data = course_rp_data, chains = 8, cores = 8,
                            iter = 2000, warmup = 1000, pars = c("alpha", "gamma", "start_expectation",
                                                                 "sigma", "log_lik", "y_rep"),
                            include = TRUE)

# Save model
write_rds(course_rp_fit, file.path(out_path, "course_rp_fit.rds"), compress = "gz")

# Remove model from memory
rm(course_rp_fit)
gc()

# Fit fine granularity model
fine_grain_fit <- sampling(fine_grain, data = fine_data, chains = 8, cores = 8,
                            iter = 2000, warmup = 1000, pars = c("alpha", "sigma", "start_expectation",
                                                                 "log_lik", "y_rep"),
                            include = TRUE)

# Save model
write_rds(fine_grain_fit, file.path(out_path, "fine_grain_fit.rds"), compress = "gz")

# Remove model from memory
rm(fine_grain_fit)
gc()

# Fit better fine granularity model
fine_grain_better_fit <- sampling(fine_grain_better, data = fine_rp_data , chains = 8, cores = 8,
                            iter = 2000, warmup = 1000, pars = c("alpha", "sigma", "gamma", "start_expectation",
                                                                 "log_lik", "y_rep"),
                            include = TRUE)

# Save model
write_rds(fine_grain_better_fit, file.path(out_path, "fine_grain_better_fit.rds"), compress = "gz")

# Remove model from memory
rm(fine_grain_better_fit)
gc()

# Fit fine RP model
fine_rp_fit <- sampling(fine_rp, data = fine_rp_data, chains = 8, cores = 8,
                            iter = 2000, warmup = 1000, pars = c("alpha", "gamma", "start_expectation",
                                                                 "sigma", "log_lik", "y_rep"),
                            include = TRUE)

# Save model
write_rds(fine_rp_fit, file.path(out_path, "fine_rp_fit.rds"), compress = "gz")

# Put system into hybernation
#system("shutdown -h")

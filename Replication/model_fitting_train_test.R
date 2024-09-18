rm(list = objects())
gc()
library(tidyverse)
library(lmerTest)
library(rstan)
library(readr)
set.seed(123)
rstan_options(auto_write = TRUE)

# Define function to create the train and test set
create_stratified_split <- function(data, sim_matrix, test_percent = 0.2) {
  # Omit NAs
  d <- na.omit(data)
  
  # Process items and sim_matrix
  item_data <- d %>%
    select(item_number, mean_population) %>%
    arrange(item_number) %>%
    unique()
  item_data$item_number <- as.integer(item_data$item_number)
  
  # Subset similarity matrix to only the items numbers actually in the data
  sim_mat <- sim_matrix[item_data$item_number, item_data$item_number]
  
  # Rename item_number
  old_names <- sort(item_data$item_number)
  new_names <- 1:length(item_data$item_number)
  
  # Create a named vector for the mapping
  item_number_map <- setNames(new_names, old_names)
  
  # Rename in data
  d$item_number <- as.integer(item_number_map[as.character(d$item_number)])
  
  n_participants <- length(unique(d$participant))
  n_items <- length(unique(d$item_number))
  
  # Sample target_test_size of participants, profiles, and items
  test_participants <- sample(unique(d$participant), size = ceiling(n_participants * test_percent))
  
  # Subset to train and test
  train <- d %>%
    filter(!(participant %in% test_participants))
  test <- d %>%
    filter(participant %in% test_participants)
  
  # Create new participant IDs for train and test sets
  train <- train %>%
    mutate(participant = as.integer(factor(participant, levels = unique(participant))))
  
  test <- test %>%
    mutate(participant = as.integer(factor(participant, levels = unique(participant))))
  
  # Ensure profiles go from 1 to the number of profiles for each set
  train <- train %>%
    mutate(profile = as.integer(factor(profile, levels = unique(profile))))
  
  test <- test %>%
    mutate(profile = as.integer(factor(profile, levels = unique(profile))))
  
  cat("Data loss (in %): ", (nrow(d) - (nrow(train) + nrow(test))) / nrow(d) * 100, "\n")
  cat("Train set size:", nrow(train), "\n")
  cat("Test set size:", nrow(test), "\n")
  cat("Actual percent size of test set:", nrow(test) / nrow(d) * 100, "\n")
  cat("Train set participants:", length(unique(train$participant)), "\n")
  cat("Test set participants:", length(unique(test$participant)), "\n")
  
  return(list(train = train, test = test, mat = sim_mat))
}

# Set path to save fitted models
out_path <- "Replication/fitted_models_lisa_split"

# Create path if it does not exist
if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

# Load data
d <- read_rds("Replication/Data/lisa/lisa_data.rds")
sim_mat <- read_rds("Replication/Data/lisa/similarity_matrix.rds")

# Turn NaN into NA
d$expectation[is.nan(d$expectation)] <- NA
d$mean_population[is.nan(d$mean_population)] <- NA

# Subset data
subsets <- list("hc_on_hc" = subset(d, type == "HC" & profile_type == "healthy"),
                "hc_on_bpd" = subset(d, type == "HC" & profile_type == "bpd"),
                "bpd_on_hc" = subset(d, type == "BPD" & profile_type == "healthy"),
                "bpd_on_bpd" = subset(d, type == "BPD" & profile_type == "bpd"))

# Create full path to save models (combination of out_path and names of subsets)
out_paths <- file.path(out_path, names(subsets))

# Create paths if they do not exist
for (path in out_paths) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

# Compile models
no_learning <- stan_model("Replication/Stan_train_test//no_learning.stan")
course_gran <- stan_model("Replication/Stan_train_test/course_gran.stan")
course_rp <- stan_model("Replication/Stan_train_test/course_rp.stan")
fine_grain <- stan_model("Replication/Stan_train_test/fine_grain.stan")
fine_rp <- stan_model("Replication/Stan_train_test/fine_rp.stan")

# Loop through subsets
for (i in seq_along(subsets)) {
  cat("Fitting model for subset", names(subsets)[i], "...\n")
  d <- subsets[[i]]
  
  # Create splits
  splits <- create_stratified_split(d, sim_mat, .2)
  train <- splits$train
  test <- splits$test
  sim_matrix <- splits$mat
  
  # Prepare data for Stan
  # Calculate populations means across factors
  # Train
  pop_means <- train %>%
    select(factor_number, mean_population) %>%
    group_by(factor_number) %>%
    summarize_all(mean)
  
  # Test
  pop_means_tilde <- test %>%
    select(factor_number, mean_population) %>%
    group_by(factor_number) %>%
    summarize_all(mean)
  
  # Calculate self means across factors
  # Train
  self_means <- train %>%
    select(factor_number, reference) %>%
    group_by(factor_number) %>%
    summarize_all(mean)
  
  # Test
  self_means_tilde <- test %>%
    select(factor_number, reference) %>%
    group_by(factor_number) %>%
    summarize_all(mean)
  
  # Extract variables for stan
  # Train
  P <- length(unique(train$participant))
  N <- length(unique(train$trial))
  Pr <- length(unique(train$profile))
  R <- nrow(train)
  rl_est <- train$expectation
  rl_feed <- train$feedback
  participant <- train$participant
  profile_id <- train$profile
  factor <- as.integer(train$factor_number)
  rl_pop_reference <- train$mean_population
  rl_self_reference <- train$reference
  n_factors <- length(unique(train$factor_number))
  
  # Test
  P_tilde <- length(unique(test$participant))
  N_tilde <- length(unique(test$trial))
  Pr_tilde <- length(unique(test$profile))
  R_tilde <- nrow(test)
  rl_est_tilde <- test$expectation
  rl_feed_tilde <- test$feedback
  participant_tilde <- test$participant
  profile_id_tilde <- test$profile
  factor_tilde <- as.integer(test$factor_number)
  rl_pop_reference_tilde <- test$mean_population
  rl_self_reference_tilde <- test$reference
  n_factors_tilde <- length(unique(test$factor_number))
  
  # Create rl stan data
  course_data <- list(R = R, P = P, Pr = Pr, n_F = n_factors, est = rl_est,
                      feedback = rl_feed, participant = participant, profile_id = profile_id, factor = factor,
                      R_tilde = R_tilde, P_tilde = P_tilde, Pr_tilde = Pr_tilde, 
                      n_F_tilde = n_factors_tilde, est_tilde = rl_est_tilde, feedback_tilde = rl_feed_tilde,
                      participant_tilde = participant_tilde, profile_id_tilde = profile_id_tilde, 
                      factor_tilde = factor_tilde)
  
  # Create course_rp data
  course_rp_data <- list(R = R, P = P, Pr = Pr, n_F = n_factors, est = rl_est,
                         feedback = rl_feed, participant = participant, profile_id = profile_id, factor = factor,
                         reference = rl_pop_reference, R_tilde = R_tilde, P_tilde = P_tilde,
                         Pr_tilde = Pr_tilde, n_F_tilde = n_factors_tilde, est_tilde = rl_est_tilde,
                         feedback_tilde = rl_feed_tilde, participant_tilde = participant_tilde,
                         profile_id_tilde = profile_id_tilde, factor_tilde = factor_tilde,
                         reference_tilde = rl_pop_reference_tilde)
  
  course_self_rp_data <- list(R = R, P = P, Pr = Pr, n_F = n_factors, est = rl_est,
                         feedback = rl_feed, participant = participant, profile_id = profile_id, factor = factor,
                         reference = rl_self_reference, R_tilde = R_tilde, P_tilde = P_tilde,
                         Pr_tilde = Pr_tilde, n_F_tilde = n_factors_tilde, est_tilde = rl_est_tilde,
                         feedback_tilde = rl_feed_tilde, participant_tilde = participant_tilde,
                         profile_id_tilde = profile_id_tilde, factor_tilde = factor_tilde,
                         reference_tilde = rl_self_reference_tilde)
  
  # Create fine_grain data
  # Train
  n_I <- length(unique(train$item_number))
  item <- train$item_number
  
  # Test
  n_I_tilde <- length(unique(test$item_number))
  item_tilde <- test$item_number
  
  
  fine_data <- list(R = R, N = N, P = P, Pr = Pr, n_I = n_I, est = rl_est,
                    feedback = rl_feed, participant = participant, profile_id = profile_id, 
                    item = item, sim_mat = sim_matrix, R_tilde = R_tilde, N_tilde = N_tilde, P_tilde = P_tilde,
                    Pr_tilde = Pr_tilde, n_I_tilde = n_I_tilde, est_tilde = rl_est_tilde,
                    feedback_tilde = rl_feed_tilde, participant_tilde = participant_tilde,
                    profile_id_tilde = profile_id_tilde, item_tilde = item_tilde)
  
  fine_rp_data <- list(R = R, N = N, P = P, Pr = Pr, n_I = n_I, est = rl_est,
                       feedback = rl_feed, participant = participant, profile_id = profile_id, 
                       item = item, sim_mat = sim_matrix, reference = rl_pop_reference,
                       R_tilde = R_tilde, N_tilde = N_tilde, P_tilde = P_tilde, Pr_tilde = Pr_tilde,
                       n_I_tilde = n_I_tilde, est_tilde = rl_est_tilde, feedback_tilde = rl_feed_tilde,
                       participant_tilde = participant_tilde, profile_id_tilde = profile_id_tilde,
                       item_tilde = item_tilde, reference_tilde = rl_pop_reference_tilde)
  
  fine_self_rp_data <- list(R = R, N = N, P = P, Pr = Pr, n_I = n_I, est = rl_est,
                       feedback = rl_feed, participant = participant, profile_id = profile_id, 
                       item = item, sim_mat = sim_matrix, reference = rl_self_reference,
                       R_tilde = R_tilde, N_tilde = N_tilde, P_tilde = P_tilde, Pr_tilde = Pr_tilde,
                       n_I_tilde = n_I_tilde, est_tilde = rl_est_tilde, feedback_tilde = rl_feed_tilde,
                       participant_tilde = participant_tilde, profile_id_tilde = profile_id_tilde,
                       item_tilde = item_tilde, reference_tilde = rl_self_reference_tilde)
  
  
  # Create data list
  no_learning_data <- list(R = R, P = P, est = rl_est, reference = rl_pop_reference, participant = participant,
                           R_tilde = R_tilde, P_tilde = P_tilde, est_tilde = rl_est_tilde, reference_tilde = rl_pop_reference_tilde,
                           participant_tilde = participant_tilde)
  
  no_learning_self_data <- list(R = R, P = P, est = rl_est, reference = rl_self_reference, participant = participant,
                           R_tilde = R_tilde, P_tilde = P_tilde, est_tilde = rl_est_tilde, reference_tilde = rl_self_reference_tilde,
                           participant_tilde = participant_tilde)

  # Run no learning stan model
  no_learning_fit <- sampling(no_learning, data = no_learning_data, chains = 8, cores = 8,
                              iter = 200, pars = c("b0", "b1", "sigma", "y_rep", "log_lik", "ae"),
                              include = TRUE)

  # Save model
  write_rds(no_learning_fit, file.path(out_paths[i], "no_learning_fit.rds"), compress = "gz")

  # Remove model from memory
  rm(no_learning_fit)
  gc()
  
  # Fit course granularity model
  course_gran_fit <- sampling(course_gran, data = course_data, chains = 8, cores = 8,
                              iter = 200, pars = c("alpha", "sigma",
                                                                   "start_expectation", "log_lik", "y_rep", "ae"),
                              include = TRUE)

  # Save model
  write_rds(course_gran_fit, file.path(out_paths[i], "course_gran_fit.rds"), compress = "gz")

  # Remove model from memory
  rm(course_gran_fit)
  gc()
  
  # Fit course RP model
  course_rp_fit <- sampling(course_rp, data = course_rp_data, chains = 8, cores = 8,
                            iter = 200, pars = c("alpha", "gamma", "start_expectation",
                                                                 "sigma", "log_lik", "y_rep", "ae"),
                            include = TRUE)

  # Save model
  write_rds(course_rp_fit, file.path(out_paths[i], "course_rp_fit.rds"), compress = "gz")

  # Remove model from memory
  rm(course_rp_fit)
  gc()
  
  # Fit fine granularity model
  fine_grain_fit <- sampling(fine_grain, data = fine_data, chains = 8, cores = 8,
                             iter = 200, pars = c("alpha", "sigma", "start_expectation",
                                                                  "log_lik", "y_rep", "ae"),
                             include = TRUE)

  # Save model
  write_rds(fine_grain_fit, file.path(out_paths[i], "fine_grain_fit.rds"), compress = "gz")

  # Remove model from memory
  rm(fine_grain_fit)
  gc()
  
  # Fit fine RP model
  fine_rp_fit <- sampling(fine_rp, data = fine_rp_data, chains = 8, cores = 8,
                          iter = 200, pars = c("alpha", "gamma", "start_expectation",
                                                               "sigma", "log_lik", "y_rep", "ae"),
                          include = TRUE)

  # Save model
  write_rds(fine_rp_fit, file.path(out_paths[i], "fine_rp_fit.rds"), compress = "gz")
  rm(fine_rp_fit)
  gc()
  
  # Fit the no learning and the two rp models with self reference points instead
  no_learning_self_fit <- sampling(no_learning, data = no_learning_self_data, chains = 8, cores = 8,
                              iter = 200, pars = c("b0", "b1", "sigma", "y_rep", "log_lik", "ae"),
                              include = TRUE)
  
  # Save model
  write_rds(no_learning_self_fit, file.path(out_paths[i], "no_learning_self_fit.rds"), compress = "gz")
  rm(no_learning_self_fit)
  gc()
  
  course_self_rp_fit <- sampling(course_rp, data = course_self_rp_data, chains = 8, cores = 8,
                            iter = 200, pars = c("alpha", "gamma", "start_expectation",
                                                                 "sigma", "log_lik", "y_rep", "ae"),
                            include = TRUE)
  
  # Save model
  write_rds(course_self_rp_fit, file.path(out_paths[i], "course_self_rp_fit.rds"), compress = "gz")
  rm(course_self_rp_fit)
  gc()
  
  fine_self_rp_fit <- sampling(fine_rp, data = fine_self_rp_data, chains = 8, cores = 8,
                          iter = 200, pars = c("alpha", "gamma", "start_expectation",
                                                               "sigma", "log_lik", "y_rep", "ae"),
                          include = TRUE)
  
  # Save model
  write_rds(fine_self_rp_fit, file.path(out_paths[i], "fine_self_rp_fit.rds"), compress = "gz")
  
  cat("--------------------\n")
}

# Put system into hybernation
#system("shutdown -h")
rm(list = objects())
gc()
library(tidyverse)
library(readr)
source("Replication/modeling_funs.R")
set.seed(123)

# Set path to save fitted models
out_path <- "Replication/fitted_models_lisa"

# Create path if it does not exist
if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

# Load data
d <- read_rds("Replication/Data/lisa/lisa_data.rds")
sim_mat <- read_rds("Replication/Data/lisa/similarity_matrix.rds")

# Ensure that profiles go from 1 to the number of profiles
d$profile <- d$profile - min(d$profile) + 1

# Turn NaN into NA
d$expectation[is.nan(d$expectation)] <- NA
d$mean_population[is.nan(d$mean_population)] <- NA

# Omit NAs
d <- na.omit(d)

# Define aic and bic functions
aic <- function(liks, n_params) {
  return(-2 * sum(liks) + 2 * n_params)
}

bic <- function(liks, n_params, n_obs) {
  return(-2 * sum(liks) + n_params * log(n_obs))
}

#bic <- function(liks, n_params, n_obs) {
#  return(n_obs * log(sum(liks) / n_obs) + n_params * log(n_obs))
#}

# Define function to fit all participants using optim
fit_all <- function(data, sim_mat) {
  # Extract unique participants
  participants <- unique(data$participant)
  n_models <- 2
  
  # Initialize matrix to store alpha estimates
  param_estimates <- matrix(NA, nrow = length(participants)*n_models, ncol = 3)
  colnames(param_estimates) <- c("model", "participant", "nll")
  
  lower_bounds <- c(0, 1)
  upper_bounds <- c(1, 8)
  
  # Fit all participants
  for (i in seq_along(participants)) {
    # Extract data for current participant
    current_part <- participants[i]
    part_data <- data %>% filter(participant == current_part)
    
    # Fit models
    no_learning_model <- lm(expectation ~ mean_population, part_data)
    no_learning_nll <- logLik(no_learning_model)
    no_learning_nll <- bic(no_learning_nll, 2, nrow(part_data))
    
    # Sanitize liks
    #no_learning_liks[is.infinite(no_learning_liks)] <- -1e6
    #no_learning_liks[is.na(no_learning_liks)] <- -1e6
    
    # Sum and negate
    #no_learning_nll <- -sum(no_learning_liks)
    #no_learning_nll <- sum(no_learning_model$residuals^2)
    #no_learning_nll <- bic(no_learning_model$residuals^2, 2, nrow(part_data))
    
    fine_gran_results <- optim(par = c(0.3, 5), fn = fit_fine_gran, part_data = part_data, sim_mat = sim_mat,
                         lower = lower_bounds, upper = upper_bounds, method = "L-BFGS-B", 
                         control = list(maxit = 20000))
    
    # Extract nll
    nll <- fine_gran_results$value
    
    # Store results
    param_estimates[i, ] <- c("no_learning", current_part, no_learning_nll)
    param_estimates[i + length(participants), ] <- c("fine_gran", current_part, nll)
  }
  
  # Return results
  return(as.data.frame(param_estimates))
}

# Run the fitting
param_estimates <- fit_all(d, sim_mat)

param_estimates$nll <- as.numeric(param_estimates$nll)

# Subset to models
no_learning_estimates <- param_estimates %>% filter(model == "no_learning")
fine_gran_estimates <- param_estimates %>% filter(model == "fine_gran")

param_estimates %>%
  select(-participant) %>%
  group_by(model) %>%
  summarise_all(sum) %>%
  ungroup()

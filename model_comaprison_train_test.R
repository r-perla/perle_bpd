rm(list = objects())
gc()
library(tidyverse)
library(rstan)
library(readr)
set.seed(123)

# Define function to compute pointwise predictive density
compute_ppd <- function(log_lik) {
  M <- nrow(log_lik)  # Number of posterior draws
  N <- ncol(log_lik)  # Number of observations in the test set
  
  ppd <- numeric(N)
  
  for (n in 1:N) {
    max_log_lik <- max(log_lik[, n])
    log_p <- -log(M) + max_log_lik + log(sum(exp(log_lik[, n] - max_log_lik)))
    ppd[n] <- exp(log_p)
  }
  
  return(ppd)
}

# Define function to re-create the train and test sets
create_stratified_split <- function(data, sim_matrix, test_percent = 0.2) {
  # Omit NAs
  d <- na.omit(data)
  
  # Sample target_test_size of participants, profiles, and items
  test_participants <- sample(unique(d$participant), size = ceiling(n_participants * test_percent))
  
  # Subset to train and test
  train <- d %>%
    filter(!(participant %in% test_participants))
  test <- d %>%
    filter(participant %in% test_participants)
  
  return(list(train = train, test = test))
}


# Set paths
path_to_models <- "Replication/fitted_models_lisa_split"
out_path <- "Replication/train_test_results"

# Create output path if it does not exist
if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

groups <- list.files(path_to_models, full.names = F)
groups <- groups[-length(groups)]
models <- list.files(file.path(path_to_models, groups[1]), full.names = F)

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

# Initialize empty storage matrix for LOOIC
loo_results_objects <- list()
looic <- matrix(NA, nrow = length(groups) * length(models), ncol = 4)
colnames(looic) <- c("group", "model", "looic", "mae")

# Loop through groups
for (i in seq_along(groups)) {
  cat("Currently processing group: ", i, " out of ", length(groups), "...\n", sep = "")
  loo_objects <- list()
  current_group <- groups[i]
  
  current_split <- create_stratified_split(subsets[[current_group]], sim_mat, .2)
  current_test <- split$test
  
  # Loop through models
  for (j in seq_along(models)) {
    current_model <- models[j]
    
    # Construct path to model file
    model_path <- file.path(path_to_models, current_group, current_model)
    
    # Read model
    current_model_fit <- read_rds(model_path)
    
    # Get predictions, target and likelihood
    preds <- extract(current_model_fit, pars = "y_pred")$y_pred
    target <- current_test$expectation
    log_lik <- extract(current_model_fit, pars = "log_lik")$log_lik
    
    # Calculate LOOIC
    ppd <- compute_ppd(log_lik)
    elpd_estimate <- mean(log(ppd))
    looic <- -2 * elpd_estimate
    
    # Calculate MAE
    mae <- mean(apply(preds, 2, function(x) mean(abs(x - target))))
    
    # Remove model from memory
    rm(current_model_fit)
    gc()
    
    # Store everything in the matrix
    looic[(i - 1) * length(models) + j, ] <- c(current_group, current_model, looic, mae)
  }
  
  current_group_result <- loo_compare(loo_objects[[models[1]]], loo_objects[[models[2]]], 
                                      loo_objects[[models[3]]], loo_objects[[models[4]]], 
                                      loo_objects[[models[5]]], loo_objects[[models[6]]])
  
  loo_results_objects[[current_group]] <- current_group_result
}

# Convert looic matrix to data frame
looic <- as.data.frame(looic)
looic$se <- as.numeric(looic$se)
looic$looic <- as.numeric(looic$looic)

# Save results
write_rds(loo_results_objects, paste0(out_path, "/model_comparisons_lisa.rds"), compress = "gz")
write_rds(looic, paste0(out_path, "/looic_lisa.rds"), compress = "gz")
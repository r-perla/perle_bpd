# Define fine granularity model function
fit_fine_gran <- function(params, part_data, sim_mat) {
  # Extract params
  alpha <- params[1]
  initial_exp <- params[2]
  
  # Subset matrix (rows and columns) to item numbers
  occuring_items <- unique(part_data$item_number)
  sim_mat <- sim_mat[occuring_items, occuring_items]
  
  # Create a vector of unique item numbers
  unique_items <- sort(unique(part_data$item_number))
  
  # Create a named vector for mapping old to new item numbers
  item_map <- setNames(seq_along(unique_items), unique_items)
  
  # Add a new column with normalized item numbers
  part_data$normalized_item_number <- item_map[as.character(part_data$item_number)]
  
  part_data$item_number <- part_data$normalized_item_number
  
  # Prepare data
  n_profiles <- 6
  n_items <- length(unique(part_data$item_number))
  n_trials <- nrow(part_data)
  
  # Initialize matrices
  expectations <- matrix(initial_exp, nrow = n_profiles, ncol = n_items)
  estimates <- rep(NA, n_trials)
  
  for (i in 1:n_trials) {
    # Extract data
    current_profile <- part_data$profile[i]
    current_item <- part_data$item_number[i]
    current_feedback <- part_data$feedback[i]
    item_sim_col <- sim_mat[, current_item]
    
    # Generate estimate
    estimates[i] <- expectations[current_profile, current_item]
    
    # Perform update
    expectations[current_profile, ] <- expectations[current_profile, ] + 
      alpha * item_sim_col * (current_feedback - expectations[current_profile, ])
  }
  
  # Calcualte residuals
  residuals <- part_data$expectation - estimates
  
  # Calculate sigma
  sigma <- sqrt(sum(residuals^2) / n_trials)
  
  # Calculate likelihood
  lik <- sum(dnorm(x = estimates, mean = part_data$expectation, sd = sigma, log = TRUE))
  
  # Sanitize likelihood
  if (is.infinite(lik) | is.na(lik)) {
    lik <- -1e6
  }
  
  bic <- -2 * lik + 2 * log(n_trials)
  return(bic)
}
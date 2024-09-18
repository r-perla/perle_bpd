data {
  // Train
  int<lower=1> P; // Number of participants
  int<lower=1> Pr; // Number of profiles
  int<lower=1> n_I; // Number of items
  int<lower=1> R; // Global number of rows in the data
  array[R] real<lower=1, upper=8> est; // Participant estimates (expectations)
  array[R] real<lower=1, upper=8> feedback; // Feedback (reward in standard RL)
  array[R] int<lower=1, upper=P> participant; // Participant ID for each row
  array[R] int<lower=1, upper=Pr> profile_id; // Profile ID for each row
  array[R] int<lower=1, upper=n_I> item; // Item ID for each row
  matrix<lower=-1, upper=1>[n_I, n_I] sim_mat; // similarity matrix
  
  // Test
  int<lower=1> P_tilde; // Number of participants
  int<lower=1> Pr_tilde; // Number of profiles
  int<lower=1> n_I_tilde; // Number of items
  int<lower=1> R_tilde; // Global number of rows in the data
  array[R_tilde] real<lower=1, upper=8> est_tilde; // Participant estimates (expectations)
  array[R_tilde] real<lower=1, upper=8> feedback_tilde; // Feedback (reward in standard RL)
  array[R_tilde] int<lower=1, upper=P_tilde> participant_tilde; // Participant ID for each row
  array[R_tilde] int<lower=1, upper=Pr_tilde> profile_id_tilde; // Profile ID for each row
  array[R_tilde] int<lower=1, upper=n_I_tilde> item_tilde; // Item ID for each row
}

parameters {
  // Population-level parameters
  real mu_alpha_raw;
  real<lower=0> sigma_alpha_raw;
  real mu_start_expectation_raw;
  real<lower=0> sigma_start_expectation_raw;
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;

  // Participant-level parameters (non-centered)
  vector[P] alpha_raw;
  vector[P] start_expectation_raw;
  vector[P] log_sigma_raw;
}

transformed parameters {
  vector<lower=0, upper=1>[P] alpha;
  vector<lower=1, upper=8>[P] start_expectation;
  vector<lower=0>[P] sigma;

  // Transformations to original scale
  alpha = inv_logit(mu_alpha_raw + sigma_alpha_raw * alpha_raw);
  start_expectation = 1 + 7 * inv_logit(mu_start_expectation_raw + sigma_start_expectation_raw * start_expectation_raw);
  sigma = exp(mu_log_sigma + sigma_log_sigma * log_sigma_raw);
}

model {
  // Priors for population-level parameters
  mu_alpha_raw ~ normal(0, 1);
  sigma_alpha_raw ~ normal(0, 0.5);
  mu_start_expectation_raw ~ normal(0, 1);
  sigma_start_expectation_raw ~ normal(0, 0.5);
  mu_log_sigma ~ normal(0, 1);
  sigma_log_sigma ~ normal(0, 0.5);

  // Priors for participant-level raw parameters
  alpha_raw ~ normal(0, 1);
  start_expectation_raw ~ normal(0, 1);
  log_sigma_raw ~ normal(0, 1);
  
  {
    // Initialize model estimates
    array[P, Pr] vector[n_I] est_model;
    for (p in 1:P) {
      for (pr in 1:Pr) {
        est_model[p, pr] = rep_vector(start_expectation[p], n_I);
      }
    }
    
    // Loop through rows
    for (r in 1:R) {
      int p = participant[r];
      int pr = profile_id[r];
      int i = item[r];
      vector[n_I] item_sim_col = col(sim_mat, i);
      
      // Likelihood
      target += normal_lpdf(est[r] | est_model[p, pr][i], sigma[p]);

      // Update model estimate
      est_model[p, pr] += alpha[p] * (feedback[r] - est_model[p, pr][i]) * item_sim_col;
    }
  }
}

generated quantities {
  array[R_tilde] real log_lik;
  array[R_tilde] real y_rep;
  vector[P_tilde] alpha_tilde = rep_vector(inv_logit(mu_alpha_raw), P_tilde); // Population-level alpha for test data
  vector[P_tilde] start_expectation_tilde = rep_vector(1 + 7 * inv_logit(mu_start_expectation_raw), P_tilde); // Population-level start_expectation for test data
  vector[P_tilde] sigma_tilde = rep_vector(exp(mu_log_sigma), P_tilde); // Population-level sigma for test data
  vector[R_tilde] ae; // Absolute error for model comparison
  
  {
    // Initialize model estimates
    array[P_tilde, Pr_tilde] vector[n_I_tilde] est_model;
    for (p in 1:P_tilde) {
      for (pr in 1:Pr_tilde) {
        est_model[p, pr] = rep_vector(start_expectation_tilde[p], n_I_tilde);
      }
    }
    
    // Loop through rows
    for (r in 1:R_tilde) {
      int p = participant_tilde[r];
      int pr = profile_id_tilde[r];
      int i = item_tilde[r];
      vector[n_I_tilde] item_sim_col = col(sim_mat, i);
      
      // Compute log likelihood
      log_lik[r] = normal_lpdf(est_tilde[r] | est_model[p, pr][i], sigma_tilde[p]);
      
      // Compute mean absolute error
      ae[r] = abs(est_tilde[r] - est_model[p, pr][i]);
      
      // Generate prediction
      y_rep[r] = fmin(fmax(normal_rng(est_model[p, pr][i], sigma_tilde[p]), 1), 8);

      // Update model estimate
      est_model[p, pr] += alpha_tilde[p] * (feedback_tilde[r] - est_model[p, pr][i]) * item_sim_col;
    }
  }
}

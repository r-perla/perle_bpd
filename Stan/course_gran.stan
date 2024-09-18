data {
  int<lower=1> P;  // Number of participants
  int<lower=1> Pr;  // Number of profiles
  int<lower=1> n_F;  // Number of factors
  int<lower=1> R;  // Global number of rows in the data
  array[R] real<lower=1, upper=8> est;  // Participant estimates (expectations)
  array[R] real<lower=1, upper=8> feedback;  // Feedback (reward in standard RL)
  array[R] int<lower=1, upper=P> participant;  // Participant ID for each row
  array[R] int<lower=1, upper=Pr> profile_id;  // Profile ID for each row
  array[R] int<lower=1, upper=n_F> factor;  // Factor ID for each row
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
    array[P, Pr, n_F] real est_model;
    for (p in 1:P) {
      for (pr in 1:Pr) {
        for (f in 1:n_F) {
          est_model[p, pr, f] = start_expectation[p];
        }
      }
    }
    
    // Loop through rows
    for (r in 1:R) {
      int p = participant[r];
      int pr = profile_id[r];
      int f = factor[r];
      
      // Likelihood
      target += normal_lpdf(est[r] | est_model[p, pr, f], sigma[p]);

      // Update model estimate
      est_model[p, pr, f] += alpha[p] * (feedback[r] - est_model[p, pr, f]);
    }
  }
}

generated quantities {
  array[R] real log_lik;
  array[R] real y_rep;
  
  {
    // Initialize model estimates
    array[P, Pr, n_F] real est_model;
    for (p in 1:P) {
      for (pr in 1:Pr) {
        for (f in 1:n_F) {
          est_model[p, pr, f] = start_expectation[p];
        }
      }
    }
    
    // Loop through rows
    for (r in 1:R) {
      int p = participant[r];
      int pr = profile_id[r];
      int f = factor[r];
      
      // Compute log likelihood
      log_lik[r] = normal_lpdf(est[r] | est_model[p, pr, f], sigma[p]);
      
      // Generate prediction
      y_rep[r] = normal_rng(est_model[p, pr, f], sigma[p]);
      y_rep[r] = fmin(fmax(y_rep[r], 1), 8);  // Ensure predictions stay within [1, 8]
      
      // Update model estimate
      est_model[p, pr, f] += alpha[p] * (feedback[r] - est_model[p, pr, f]);
    }
  }
}

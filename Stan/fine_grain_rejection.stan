data {
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
}

parameters {
  vector<lower=0, upper=1>[P] alpha; // Learning rates
  vector[n_I] start_expectation; // Item-level initial expectations (unbounded)
  vector[P] bias; // Individual scale location shift
  vector<lower=0, upper=1>[P] mu_beta; // Mean of beta distribution for each participant
  vector<lower=2>[P] nu_beta; // Concentration of beta distribution for each participant
  real<lower=1, upper=8> global_location; // Global location parameter for bounding
  real<lower=0> bias_sd; // SD for bias
  real<lower=0> mu_beta_sd; // SD for mu_beta
  real<lower=0> nu_beta_sd; // SD for nu_beta
  real<lower=0> start_expectation_sd; // SD for start_expectation
}

transformed parameters {
  vector<lower=0>[P] alpha_beta;
  vector<lower=0>[P] beta_beta;
  
  for (p in 1:P) {
    alpha_beta[p] = mu_beta[p] * nu_beta[p];
    beta_beta[p] = (1 - mu_beta[p]) * nu_beta[p];
  }
}

model {
  // Priors
  alpha ~ beta(2, 3);  // Slightly biased towards lower learning rates
  start_expectation ~ normal(4.5, start_expectation_sd);  // Centered around middle of scale
  bias ~ normal(0, bias_sd);
  mu_beta ~ normal(0.68, mu_beta_sd);  // Centered slightly above 0.5 to reflect right skew
  nu_beta ~ gamma(10, 0.5);  // Encourages values around 20, adjusting the concentration
  global_location ~ normal(4.5, 0.5);  // Tighter prior around middle of scale
  
  // Hyperpriors
  bias_sd ~ normal(0, 0.5) T[0,];
  mu_beta_sd ~ normal(0, 0.1) T[0,];  // Tighter SD to keep means relatively close
  nu_beta_sd ~ normal(0, 5) T[0,];  // Allows for some variation in concentration
  start_expectation_sd ~ normal(0, 1) T[0,];

  {
    // Initialize model estimates (unbounded)
    array[P, Pr] vector[n_I] est_model;
    for (p in 1:P) {
      for (pr in 1:Pr) {
        est_model[p, pr] = start_expectation;
      }
    }
    
    // Loop through rows
    for (r in 1:R) {
      int p = participant[r];
      int pr = profile_id[r];
      int i = item[r];
      vector[n_I] item_sim_col = col(sim_mat, i);
      
      real unbounded_est = est_model[p, pr][i] + bias[p];
      real bounded_est = 1 + 7 * beta_cdf(inv_logit(unbounded_est) | alpha_beta[p], beta_beta[p]);
      
      // Likelihood
      target += beta_lpdf((est[r] - 1) / 7 | alpha_beta[p], beta_beta[p]);

      // Update model estimate (unbounded) with original error calculation
      est_model[p, pr] += alpha[p] * (feedback[r] - bounded_est) * item_sim_col;
    }
  }
}

generated quantities {
  array[R] real log_lik;
  array[R] real y_rep;
  
  {
    // Initialize model estimates (unbounded)
    array[P, Pr] vector[n_I] est_model;
    for (p in 1:P) {
      for (pr in 1:Pr) {
        est_model[p, pr] = start_expectation;
      }
    }
    
    // Loop through rows
    for (r in 1:R) {
      int p = participant[r];
      int pr = profile_id[r];
      int i = item[r];
      vector[n_I] item_sim_col = col(sim_mat, i);
      
      real unbounded_est = est_model[p, pr][i] + bias[p];
      real bounded_est = 1 + 7 * beta_cdf(inv_logit(unbounded_est) | alpha_beta[p], beta_beta[p]);
      
      // Compute log likelihood
      log_lik[r] = beta_lpdf((est[r] - 1) / 7 | alpha_beta[p], beta_beta[p]);
      
      // Generate prediction
      y_rep[r] = 1 + 7 * beta_rng(alpha_beta[p], beta_beta[p]);

      // Update model estimate (unbounded) with original error calculation
      est_model[p, pr] += alpha[p] * (feedback[r] - bounded_est) * item_sim_col;
    }
  }
}

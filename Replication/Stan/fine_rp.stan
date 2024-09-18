data {
  int<lower=1> P;  // Number of participants
  int<lower=1> Pr; // Number of profiles
  int<lower=1> n_I; // Number of items
  int<lower=1> R;  // Global number of rows in the data
  array[R] real est;  // Participant estimates (expectations)
  array[R] real feedback;  // Feedback (reward in standard RL)
  array[R] int<lower=1, upper=P> participant;  // Participant ID for each row
  array[R] int<lower=1, upper=Pr> profile_id;  // Profile ID for each row
  array[R] int<lower=1, upper=n_I> item;  // Item ID for each row
  array[R] real<lower=1, upper=8> reference;  // Reference value for each trait
  matrix<lower=-1, upper=1>[n_I, n_I] sim_mat;  // Similarity matrix between items
}

parameters {
  // Population-level parameters
  real mu_alpha_raw;
  real<lower=0> sigma_alpha_raw;
  real mu_gamma_raw;
  real<lower=0> sigma_gamma_raw;
  real mu_start_expectation_raw;
  real<lower=0> sigma_start_expectation_raw;
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;

  // Participant-level parameters (non-centered)
  vector[P] alpha_raw;
  vector[P] gamma_raw;
  vector[P] start_expectation_raw;
  vector[P] log_sigma_raw;
}

transformed parameters {
  vector<lower=0, upper=1>[P] alpha;
  vector<lower=0, upper=1>[P] gamma;
  vector<lower=1, upper=8>[P] start_expectation;
  vector<lower=0>[P] sigma;

  // Transformations to original scale
  alpha = inv_logit(mu_alpha_raw + sigma_alpha_raw * alpha_raw);
  gamma = inv_logit(mu_gamma_raw + sigma_gamma_raw * gamma_raw);
  start_expectation = 1 + 7 * inv_logit(mu_start_expectation_raw + sigma_start_expectation_raw * start_expectation_raw);
  sigma = exp(mu_log_sigma + sigma_log_sigma * log_sigma_raw);
}

model {
  // Priors for population-level parameters
  mu_alpha_raw ~ normal(0, 1);
  sigma_alpha_raw ~ normal(0, 0.5);
  mu_gamma_raw ~ normal(0, 1);
  sigma_gamma_raw ~ normal(0, 0.5);
  mu_start_expectation_raw ~ normal(0, 1);
  sigma_start_expectation_raw ~ normal(0, 0.5);
  mu_log_sigma ~ normal(0, 1);
  sigma_log_sigma ~ normal(0, 0.5);

  // Priors for participant-level raw parameters
  alpha_raw ~ normal(0, 1);
  gamma_raw ~ normal(0, 1);
  start_expectation_raw ~ normal(0, 1);
  log_sigma_raw ~ normal(0, 1);

  // Model block
  {
    // Initialize model estimates
    array[P, Pr] vector[n_I] est_model;
    for (p in 1:P) {
      for (pr in 1:Pr) {
        est_model[p, pr] = rep_vector(start_expectation[p], n_I);
      }
    }
    
    // Loop through rows of data
    for (r in 1:R) {
      int p = participant[r];
      int pr = profile_id[r];
      int i = item[r];
      real ref = reference[r];
      vector[n_I] item_sim_col = col(sim_mat, i);
      
      // Generate current prediction as weighted sum of reference and model estimate
      real pred = gamma[p] * ref + (1 - gamma[p]) * est_model[p, pr][i];
      
      // Likelihood: participant's estimate follows normal distribution around prediction
      est[r] ~ normal(pred, sigma[p]);

      // Update model estimate using similarity-based generalization
      est_model[p, pr] += alpha[p] * (feedback[r] - est_model[p, pr][i]) * item_sim_col;
    }
  }
}

generated quantities {
  array[R] real log_lik;  // Log likelihood for each observation
  array[R] real y_rep;  // Posterior predictive samples
  
  {
    // Initialize model estimates (same as in model block)
    array[P, Pr] vector[n_I] est_model;
    for (p in 1:P) {
      for (pr in 1:Pr) {
        est_model[p, pr] = rep_vector(start_expectation[p], n_I);
      }
    }
    
    // Loop through rows of data
    for (r in 1:R) {
      int p = participant[r];
      int pr = profile_id[r];
      int i = item[r];
      real ref = reference[r];
      vector[n_I] item_sim_col = col(sim_mat, i);
      
      // Generate current prediction (same as in model block)
      real pred = gamma[p] * ref + (1 - gamma[p]) * est_model[p, pr][i];
      
      // Compute log likelihood for each observation
      log_lik[r] = normal_lpdf(est[r] | pred, sigma[p]);
      
      // Generate posterior predictive sample, constrained to [1, 8]
      y_rep[r] = normal_rng(pred, sigma[p]);
      y_rep[r] = fmin(fmax(y_rep[r], 1), 8);

      // Update model estimate (same as in model block)
      est_model[p, pr] += alpha[p] * (feedback[r] - est_model[p, pr][i]) * item_sim_col;
    }
  }
}

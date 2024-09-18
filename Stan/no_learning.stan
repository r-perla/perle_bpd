data {
  int<lower=1> P;  // Number of participants
  int<lower=1> R;  // Number of observations
  vector[R] est;  // Observed estimates
  vector[R] reference;  // Reference values
  array[R] int<lower=1, upper=P> participant;  // Participant IDs
}

parameters {
  // Population-level parameters
  real mu_b0;
  real<lower=0> sigma_b0;
  real mu_b1;
  real<lower=0> sigma_b1;
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;

  // Participant-level parameters (non-centered)
  vector[P] b0_raw;
  vector[P] b1_raw;
  vector[P] log_sigma_raw;
}

transformed parameters {
  vector[P] b0 = mu_b0 + sigma_b0 * b0_raw;
  vector[P] b1 = mu_b1 + sigma_b1 * b1_raw;
  vector<lower=0>[P] sigma = exp(mu_log_sigma + sigma_log_sigma * log_sigma_raw);
}

model {
  // Priors for population-level parameters
  mu_b0 ~ normal(0, 5);
  sigma_b0 ~ normal(0, 2);
  mu_b1 ~ normal(0, 5);
  sigma_b1 ~ normal(0, 2);
  mu_log_sigma ~ normal(0, 1);
  sigma_log_sigma ~ normal(0, 1);

  // Priors for participant-level parameters
  b0_raw ~ normal(0, 1);
  b1_raw ~ normal(0, 1);
  log_sigma_raw ~ normal(0, 1);

  // Likelihood
  est ~ normal(b0[participant] + b1[participant] .* reference, sigma[participant]);
}

generated quantities {
  vector[R] log_lik;  // Log-likelihood for model comparison
  vector[R] y_rep;    // Replicated data for posterior predictive checks

  for (r in 1:R) {
    // Compute log-likelihood
    log_lik[r] = normal_lpdf(est[r] | b0[participant[r]] + b1[participant[r]] * reference[r], sigma[participant[r]]);

    // Generate replicated data
    y_rep[r] = normal_rng(b0[participant[r]] + b1[participant[r]] * reference[r], sigma[participant[r]]);

    // Ensure predictions stay within [1, 8] if necessary
    y_rep[r] = fmin(fmax(y_rep[r], 1), 8);
  }
}

rm(list = objects())
gc()
library(tidyverse)
library(jtools)
library(reshape2)
library(rstan)
library(bayesplot)
library(readr)
library(moments)
set.seed(NULL) # for reproducibility of plotting multiples

# Load real data
d <- read_rds("Replication/Data/lisa/lisa_data.rds") %>%
  na.omit()

# Load model objects
model_object <- read_rds("Replication/fitted_models_lisa/fine_rp_fit.rds")

# Extract generated data from stan objects
y_reps <- extract(model_object, pars = "y_rep")$y_rep %>%
  round()

# Select 10 random iterations
random_iterations <- sample(dim(y_reps)[1], 10)

# Extract 10 random iterations and collapse across participants
y_reps_10 <- y_reps[random_iterations, ]

# Prepare actual data
y <- d$expectation

# Combine both into one df
y_df <- data.frame(
  value = c(y, as.vector(t(y_reps_10))),
  source = factor(c(rep("actual", length(y)), rep(paste0("generated_", 1:10), each = length(y))))
)

# Calculate true mean, sd, and skewness
true_mean <- mean(y)
true_sd <- sd(y)
true_skewness <- skewness(y)

# Calculate means, sds, and skewness for all iterations
y_reps_mean <- apply(y_reps, 1, mean, na.rm = TRUE)
y_reps_sd <- apply(y_reps, 1, sd, na.rm = TRUE)
y_reps_skewness <- apply(y_reps, 1, skewness, na.rm = TRUE)

# Calculate Bayesian p-values
p_value_mean <- mean(y_reps_mean > true_mean)
p_value_sd <- mean(y_reps_sd > true_sd)
p_value_skewness <- mean(y_reps_skewness > true_skewness)

# Create dataframes for plotting
df_mean <- data.frame(value = y_reps_mean)
df_sd <- data.frame(value = y_reps_sd)
df_skewness <- data.frame(value = y_reps_skewness)

# PLOTTING ----
# Set general theme
theme_set(
  theme_apa() +
    theme(legend.position = "top",
          legend.title = element_text(size = 22, color = "#EEEEEE"),
          legend.text = element_text(size = 18, color = "#EEEEEE"),
          axis.title.x = element_text(size = 22, color = "#EEEEEE"),
          axis.title.y = element_text(size = 22, color = "#EEEEEE"),
          axis.text = element_text(size = 18, color = "#EEEEEE"),
          axis.ticks = element_line(color = "#EEEEEE"),
          panel.background = element_rect(fill = "#323232"),
          plot.background = element_rect(fill = "#1E2022"),
          legend.background = element_rect(fill = "#1E2022"),
          legend.key = element_rect(fill = "#1E2022"),
          strip.background = element_rect(fill = "#4D4D4D"),
          strip.text = element_text(size = 22, color = "#EEEEEE"),
          panel.grid.major = element_line(color = "#4D4D4D"),
          panel.grid.minor = element_line(color = "#4D4D4D"),
          plot.title = element_text(size = 24, color = "#EEEEEE"),
          plot.subtitle = element_text(size = 18, color = "#EEEEEE"),
          strip.text.x.top = element_text(size = 22, color = "#EEEEEE"))
)

# Plot histograms for actual data and 10 generated datasets
ggplot(y_df, aes(x = value, fill = source)) +
  geom_histogram(position = "identity", alpha = 1, bins = 50, color = "#1E2022") +
  scale_fill_manual(values = c("actual" = "#EEEEEE", 
                               setNames(rainbow(10), paste0("generated_", 1:10)))) +
  labs(x = "Expectation", y = "Count", fill = "Source") +
  facet_wrap(~source, scales = "free_y", ncol = 3) +
  theme(legend.position = "none")

# Plot histogram of generated means vs true mean
df_mean %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 50, color = "#1E2022", fill = "#00ADB5") +
  geom_vline(xintercept = true_mean, linetype = "dashed", color = "red") +
  labs(x = "Average Mean Expectation", y = "Count")

# Plot histogram of generated SDs vs true SD
df_sd %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 50, color = "#1E2022", fill = "#00ADB5") +
  geom_vline(xintercept = true_sd, linetype = "dashed", color = "red") +
  labs(x = "Average SD Expectation", y = "Count")

# Plot histogram of generated skewness vs true skewness
df_skewness %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 50, color = "#1E2022", fill = "#00ADB5") +
  geom_vline(xintercept = true_skewness, linetype = "dashed", color = "red") +
  labs(x = "Average Skewness Expectation", y = "Count")

# Extract alpha parameter
alphas <- colMeans(extract(model_object, pars = "alpha")$alpha)

# Plot histogram of alpha
data.frame(alpha = alphas) %>%
  ggplot(aes(x = alpha)) +
  geom_histogram(bins = 9, color = "#1E2022", fill = "#00ADB5") +
  labs(x = "Alpha", y = "Count")

# Extract gamma parameter
gammas <- colMeans(extract(model_object, pars = "gamma")$gamma)

# Plot histogram of gamma
data.frame(gamma = gammas) %>%
  ggplot(aes(x = gamma)) +
  geom_histogram(bins = 9, color = "#1E2022", fill = "#00ADB5") +
  labs(x = "Gamma", y = "Count")

rm(list = objects())
gc()
library(tidyverse)
library(jtools)
library(reshape2)
library(rstan)
library(bayesplot)
library(readr)
set.seed(NULL) # for reproducibility of plotting multiples

# Load real data
d <- read_rds("Replication/Data/experiments/experiment_one.rds")

# Load model objects
model_object <- read_rds("Replication/fitted_models/no_learning_fit.rds")

# Extract generated data from stan objects
y_reps <- extract(model_object, pars = "y_rep")$y_rep

# Select 10 random iterations
random_iterations <- sample(dim(y_reps)[1], 10)

# Extract 10 random iterations and collapse across participants
y_reps_10 <- apply(y_reps[random_iterations, , ], c(1, 2), mean, na.rm = TRUE)

# Prepare actual data
y <- d %>%
  select(trial, expectation) %>%
  group_by(trial) %>%
  summarise(expectation = mean(expectation, na.rm = TRUE))

# Combine both into one df
y_df <- data.frame(
  value = c(y$expectation, as.vector(t(y_reps_10))),
  source = factor(c(rep("actual", nrow(y)), rep(paste0("generated_", 1:10), each = nrow(y))))
)

# Calculate true mean and sd
true_mean <- mean(y$expectation)
true_sd <- sd(y$expectation)

# Calculate means and sds for all iterations
y_reps_mean <- apply(y_reps, 1, mean, na.rm = TRUE)
y_reps_sd <- apply(y_reps, 1, sd, na.rm = TRUE)

# Calculate Bayesian p-values
p_value_mean <- mean(y_reps_mean > true_mean)
p_value_sd <- mean(y_reps_sd > true_sd)

# Create dataframes for plotting
df_mean <- data.frame(value = y_reps_mean)
df_sd <- data.frame(value = y_reps_sd)

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
  geom_histogram(bins = 50, color = "#EEEEEE") +
  geom_vline(xintercept = true_mean, linetype = "dashed", color = "red")

df_sd %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 50, color = "#EEEEEE") +
  geom_vline(xintercept = true_sd, linetype = "dashed", color = "red")

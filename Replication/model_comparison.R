rm(list = objects())
gc()
library(tidyverse)
library(rstan)
library(loo)
library(readr)
options(mc.cores = 8)

# Set path to plots
out_path <- "Replication/Plots/"

# Create directory if it does not exist
if (!dir.exists(out_path)) {
  dir.create(out_path)
}

# Calculate loo
course_gran <- read_rds("Replication/fitted_models/course_gran_fit.rds")
course_gran_loo <- loo(course_gran)
rm(course_gran)
gc()

course_rp <- read_rds("Replication/fitted_models/course_rp_fit.rds")
course_rp_loo <- loo(course_rp)
rm(course_rp)
gc()

fine_gran <- read_rds("Replication/fitted_models/fine_grain_fit.rds")
fine_gran_loo <- loo(fine_gran)
rm(fine_gran)
gc()

fine_gran_better <- read_rds("Replication/fitted_models/fine_grain_better_fit.rds")
fine_gran_better_loo <- loo(fine_gran_better)
rm(fine_gran_better)
gc()

fine_rp <- read_rds("Replication/fitted_models/fine_rp_fit.rds")
fine_rp_loo <- loo(fine_rp)
rm(fine_rp)
gc()

no_learning <- read_rds("Replication/fitted_models/no_learning_fit.rds")
no_learning_loo <- loo(no_learning)
rm(no_learning)
gc()

# Compare models
result <- loo_compare(course_gran_loo, course_rp_loo, fine_gran_loo, 
                      fine_gran_better_loo, fine_rp_loo, no_learning_loo)

# Save results
write_rds(result, "Replication/model_comparison.rds", compress = "gz")

# Alternatively, load in the results
result <- read_rds("Replication/model_comparison.rds")

# Show result
result

# Construct z-test for differences in elpd_loo between the best two models
difference <- result[2, 1]
se_diff <- result[2, 2]
z <- difference / se_diff
p <- 2 * (1 - pnorm(abs(z)))
p

# Extract LOOIC from models
course_gran_looic <- course_gran_loo$estimates[3, 1]
course_gran_looic_se <- course_gran_loo$estimates[3, 2]

course_rp_looic <- course_rp_loo$estimates[3, 1]
course_rp_looic_se <- course_rp_loo$estimates[3, 2]

fine_gran_looic <- fine_gran_loo$estimates[3, 1]
fine_gran_looic_se <- fine_gran_loo$estimates[3, 2]

fine_gran_better_looic <- fine_gran_better_loo$estimates[3, 1]
fine_gran_better_looic_se <- fine_gran_better_loo$estimates[3, 2]

fine_rp_looic <- fine_rp_loo$estimates[3, 1]
fine_rp_looic_se <- fine_rp_loo$estimates[3, 2]

no_learning_looic <- no_learning_loo$estimates[3, 1]
no_learning_looic_se <- no_learning_loo$estimates[3, 2]

# Prepare for data frame
model_names <- c("course_gran_m2", "course_rp_m3", "fine_gran_m4", "fine_gran_items_m6", "fine_rp_m5", "no_learning_m1")

# Create data frame
looic_df <- data.frame(
  model = model_names,
  looic = c(course_gran_looic, course_rp_looic, fine_gran_looic, fine_gran_better_looic, fine_rp_looic, no_learning_looic),
  se = c(course_gran_looic_se, course_rp_looic_se, fine_gran_looic_se, fine_gran_better_looic_se, fine_rp_looic_se, no_learning_looic_se)
)

# Save data frame
write_rds(looic_df, "Replication/looic_df.rds", compress = "gz")

# Alternatively load in the data
looic_df <- read_rds("Replication/looic_df.rds")

looic_df$looic <- looic_df$looic - min(looic_df$looic)

# Plot barplot of LOOIC by model
ggplot(looic_df, aes(y = reorder(model, looic), x = looic)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "LOOIC by model",
       x = "Model",
       y = "LOOIC") +
  jtools::theme_apa()

# Subset and plot without m6
looic_df_no_m6 <- subset(looic_df, model != "fine_gran_better_m6")
looic_df_no_m6$looic <- looic_df_no_m6$looic - min(looic_df_no_m6$looic)

ggplot(looic_df_no_m6, aes(x = model, y = looic, ymin = looic - se, ymax = looic + se)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(width = 0.2) +
  labs(title = "LOOIC by model",
       x = "Model",
       y = "LOOIC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Add an identifyer to both of these dfs
looic_df$identifier <- "With improved model"
looic_df_no_m6$identifier <- "Without improved model"

# rbind them
looic_df_all <- rbind(looic_df, looic_df_no_m6)

# Create a named vector for mapping
model_names <- c(
  "fine_rp_m5" = "Fine Granularity RP",
  "fine_gran_m4" = "Fine Granularity",
  "fine_gran_better_m6" = "Fine Granularity (improved)",
  "course_rp_m3" = "Course Granularity RP",
  "course_gran_m2" = "Course Granularity"
)

# Add a new column to your dataframe with the new names
looic_df_all <- looic_df_all %>%
  mutate(model_name = model_names[model])

# Now use the new column in your ggplot code
ggplot(looic_df_all, aes(y = reorder(model_name, - looic - max(looic)), x = looic - max(looic))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = NULL,
       x = "Model",
       y = "LOOIC") +
  jtools::theme_apa() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~identifier, scales = "free")
ggsave(paste0(out_path, "looic_comparison.svg"), width = 12, height = 4, dpi = 300)

# Instead generate df of ELPD differences from the comparison
# Map order of models in the comparison to the model names
model_mapper <- list(model1 = "course_gran_m2", model2 = "course_rp_m3", model3 = "fine_gran_m4", 
                    model4 = "fine_gran_better_m6", model5 = "fine_rp_m5")

# Map to rownames of results
model_order <- rownames(result)

model_names <- unname(unlist(model_mapper[model_order]))

# Create data frame
elpd_diff_df <- data.frame(
  model = model_names,
  elpd_diff = result[, 1],
  se = result[, 2]
)

# Save as rds
write_rds(elpd_diff_df, "Replication/elpd_diff_df.rds", compress = "gz")

# Plot as barplot
ggplot(elpd_diff_df, aes(x = model, y = elpd_diff, ymin = elpd_diff - se, ymax = elpd_diff + se)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(width = 0.2) +
  labs(title = "ELPD differences by model",
       x = "Model",
       y = "ELPD difference") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Subset and plot without m6
elpd_diff_df_no_m6 <- subset(elpd_diff_df, model != "fine_gran_better_m6")

# Recalculate differences
elpd_diff_df_no_m6$elpd_diff <- elpd_diff_df_no_m6$elpd_diff - max(elpd_diff_df_no_m6$elpd_diff)

# Also do the same for the se using the appropriate formula
elpd_diff_df_no_m6$se <- sqrt(elpd_diff_df_no_m6$se^2 - min(elpd_diff_df_no_m6$se^2)) * 2

ggplot(elpd_diff_df_no_m6, aes(x = model, y = elpd_diff, ymin = elpd_diff - se, ymax = elpd_diff + se)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(width = 0.2) +
  labs(title = "ELPD differences by model",
       x = "Model",
       y = "ELPD difference") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Add an identifyer to both of these dfs
elpd_diff_df$identifier <- "with_m6"
elpd_diff_df_no_m6$identifier <- "without_m6"

# rbind them
elpd_diff_df_all <- rbind(elpd_diff_df, elpd_diff_df_no_m6)

# Plot again with identifier as facet
ggplot(elpd_diff_df_all, aes(x = model, y = elpd_diff, ymin = elpd_diff - se, ymax = elpd_diff + se)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(width = 0.2) +
  labs(title = "ELPD differences by model",
       x = "Model",
       y = "ELPD difference") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~identifier, scales = "free_x")

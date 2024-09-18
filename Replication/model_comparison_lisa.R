rm(list = objects())
gc()
library(tidyverse)
library(rstan)
library(loo)
library(readr)
options(mc.cores = 8)

# Set paths
path_to_models <- "Replication/fitted_models_lisa"
out_path <- "Replication"

groups <- list.files(path_to_models, full.names = F)
groups <- groups[-length(groups)]
models <- list.files(file.path(path_to_models, groups[1]), full.names = F)

# Initialize empty storage matrix for LOOIC
loo_results_objects <- list()
looic <- matrix(NA, nrow = length(groups) * length(models), ncol = 4)
colnames(looic) <- c("group", "model", "looic", "se")

# Loop through groups
for (i in seq_along(groups)) {
  cat("Currently processing group: ", i, " out of ", length(groups), "...\n", sep = "")
  loo_objects <- list()
  current_group <- groups[i]
  
  # Loop through models
  for (j in seq_along(models)) {
    current_model <- models[j]
    
    # Construct path to model file
    model_path <- file.path(path_to_models, current_group, current_model)
    
    # Read model
    current_model_fit <- read_rds(model_path)
    
    # Get loo
    loo_objects[[current_model]] <- loo(current_model_fit)
    
    # Remove model from memory
    rm(current_model_fit)
    gc()
    
    # Store everything in the matrix
    looic[(i - 1) * length(models) + j, ] <- c(current_group, current_model, loo_objects[[current_model]]$estimates[3, 1], loo_objects[[current_model]]$estimates[3, 2])
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

# Put system into hibernation
#system("shutdown -h")

# Alternatively, load in the results
results <- read_rds("Replication/model_comparisons_lisa.rds")
looic_df <- read_rds("Replication/looic_lisa.rds")

looic_df <- looic_df %>%
  group_by(group) %>%
  mutate(looic = looic - min(looic)) %>%
  ungroup()

# Plot barplot of LOOIC by model
ggplot(looic_df, aes(y = reorder(model, looic), x = looic)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  facet_wrap(~group, scales = "free") +
  labs(title = "LOOIC by model",
       x = "Model",
       y = "LOOIC") +
  jtools::theme_apa()

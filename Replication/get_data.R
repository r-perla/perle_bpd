rm(list = objects())
library(tidyverse)
library(R.matlab)
library(readr)

# Set path to output folder
out_path <- "Replication/Data/experiments"

# Create path if it does not exist
if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

# Load data
d <- readMat("Data/experiments/ExperimentOne.mat")

# Define function to clean up the mess
process_data <- function(nested_list, outer_index) {
  do.call(rbind, lapply(seq_along(nested_list), function(inner_index) {
    df <- as.data.frame(nested_list[[inner_index]])
    df$sort_version <- outer_index
    df$participant <- inner_index
    df$trial <- seq_len(nrow(df))
    return(df)
  }))
}

# Apply and combine
result_list <- lapply(seq_along(d$dataSort), function(i) process_data(d$dataSort[[i]], i))
final_df <- do.call(rbind, result_list)

# Reset row names and reorder columns
rownames(final_df) <- NULL
final_df <- final_df[, c("sort_version", "participant", "trial", 
                         setdiff(names(final_df), c("sort_version", "participant", "trial")))]

# Ensure the data frame is ordered correctly
final_df <- final_df[order(final_df$sort_version, final_df$participant, final_df$trial), ]

colnames(final_df)[4:length(colnames(final_df))] <- c('expectation', 'feedback', 'reference', 'profile', 
                                                      'reset', 'mean_population', 'item_number', 'factor_number')

# Convert sort version to factor and rename
final_df$sort_version <- factor(final_df$sort_version, levels = 1:4, 
                                labels = c("irrlevant1", "irrlevant2", "irrlevant3", "use_this"),
                                ordered = FALSE)

# Subset to use_this
final_df <- subset(final_df, sort_version == "use_this")

# Drop column
final_df <- select(final_df, -sort_version)

# Fix factor number
final_df$factor_number <- final_df$factor_number - min(final_df$factor_number) + 1

# View the result
head(final_df, 20)

# Extract similarity matrix
sim_mat <- d$sweepR

# Write to file
write_rds(final_df, file.path(out_path, "experiment_one.rds"), compress = "gz")
write_csv(final_df, file.path(out_path, "experiment_one.csv"))
write_rds(sim_mat, file.path(out_path, "similarity_matrix.rds"), compress = "gz")

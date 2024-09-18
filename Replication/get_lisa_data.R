rm(list = objects())
library(tidyverse)
library(R.matlab)
library(readr)

# Set path to output folder
out_path <- "Replication/Data/lisa"

# Create path if it does not exist
if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

# Load data
d <- readMat("Replication/lisa_data/dataClean_12-Jun-2023_1453.mat")

# Remove every second element in d$HC and d$BPD
d$HC <- d$HC[seq_along(d$HC) %% 2 != 1]
d$BPD <- d$BPD[seq_along(d$BPD) %% 2 != 1]

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
result_list_hc <- lapply(seq_along(d$HC), function(i) process_data(d$HC[[i]], i))
result_list_bpd <- lapply(seq_along(d$BPD), function(i) process_data(d$BPD[[i]], i))

hc_df <- do.call(rbind, result_list_hc)
hc_df$type <- "HC"

bpd_df <- do.call(rbind, result_list_bpd)
bpd_df$type <- "BPD"

final_df <- bind_rows(hc_df, bpd_df)

# Reset row names and reorder columns
rownames(final_df) <- NULL
final_df <- final_df[, c("sort_version", "participant", "trial", 
                         setdiff(names(final_df), c("sort_version", "participant", "trial")))]

# Swap colnames for participant and sort version
colnames(final_df)[1:2] <- c("participant", "sort_version")

# Ensure the data frame is ordered correctly
final_df <- final_df[order(final_df$sort_version, final_df$participant, final_df$trial), ]

colnames(final_df)[4:length(colnames(final_df))] <- c('expectation', 'feedback', 'reference', 'profile', 
                                                      'reset', 'mean_population', 'item_number', 'factor_number',
                                                      "type")

# Drop column
final_df <- select(final_df, -sort_version)

# Fix factor number
final_df$factor_number <- final_df$factor_number - min(final_df$factor_number) + 1

# Create new profile type column
final_df <- final_df %>%
  mutate(profile_type = case_when(
    profile %in% 1:3 ~ "bpd",
    profile %in% 4:6 ~ "healthy",
    TRUE ~ NA
  ))

# View the result
head(final_df, 20)

# Extract similarity matrix
sim_mat <- d$sweepR

# Write to file
write_rds(final_df, file.path(out_path, "lisa_data.rds"), compress = "gz")
write_rds(sim_mat, file.path(out_path, "similarity_matrix.rds"), compress = "gz")

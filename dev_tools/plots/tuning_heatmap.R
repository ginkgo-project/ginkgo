# These packages are required, to install them, use the package manager or open
# an R session and type:
# install.packages("jsonlite", "tidyr", "ggplot2", "scales")
library(jsonlite)
library(ggplot2)
library(scales)
library(tidyr)

# Manage arguments
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) {
    stop("Usage: Rscript tuning_heatmap.R input_directory output_graphics_file\n", call=FALSE)
}
input <- args[1]
output <- args[2]

# Read the input json files into a dataframe
files <- list.files(paste(input), recursive=TRUE, pattern = "*.json", full.names=TRUE)
df_tmp <- list()
count <- 1
for (i in files)
{
    tmp <-jsonlite::fromJSON(i,flatten=TRUE)
    df_tmp[[count]] <- as.data.frame(tmp)[,c("problem.name", "problem.nonzeros",
                                             "spmv.coo.time", "spmv.coo.tuning.values",
                                             "spmv.coo.tuning.time")]
    count <- count +1
}
# Merge all the separate dataframes
df_merged <- rbind_pages(df_tmp)
# Unnest the two vectors
df <- as.data.frame(unnest(df,spmv.coo.tuning.values,spmv.coo.tuning.time))
# Now that all columns are vectors, compute the speedup using vector operations
df$spmv.coo.speedup <- df$spmv.coo.time/df$spmv.coo.tuning.time

# Plot the values
ggplot(df, aes(factor(problem.nonzeros), factor(spmv.coo.tuning.values), fill=spmv.coo.speedup)) +
    geom_tile() +
    scale_fill_gradientn(
        colours=c("red", "yellow", "skyblue", "darkblue"),
        values = rescale(c(min(df$speedup),
                           1.0,
                           1.11,
                           max(df$speedup)))) +
    ggtitle("Speedup of tuned value against COO SpMV")+ xlab("nonzeros")+ ylab("tuned value (multiple)") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), plot.title = element_text(hjust=0.5))

# Save to the output file
ggsave(paste(output), width=9, height=7)

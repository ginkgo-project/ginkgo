# Simple script for evaluating timings and storage of SpMVs
# with different matrix storage formats
# Reading input files inspired by the tuning_heatmap.R

# Author: Luka Stanisic
# Email: luka.stanisic@huawei.com
# Affiliation: Huawei Technologies Duesseldorf GmbH
# License: 3-clause BSD

# Example of usage for the test data
# (test data was manually modified and it doesnot represent real scenarios)
# Rscript spmv_formats.R testdata/sys1 test

# Loaded necessary R libraries
# Suppressing warnings and messages as in R library loading
# is often quite verbose
suppressMessages(suppressWarnings(library(jsonlite)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(reshape2)))
suppressMessages(suppressWarnings(library(plyr)))

# Specify which columns to read from the input file
columns <- "problem.name|optimal.spmv|spmv.*.time|spmv.*.storage"

# Manage arguments
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2)
{
	 stop("Usage: Rscript spmv_formats.R input_directory
	      output_graphics_file\n
	      Example: Rscript spmv_formats.R testdata/sys1 test\n",
	      call=FALSE)
}
input <- args[1]
output <- args[2]

# Read input data
files <- list.files(paste(input), recursive=TRUE, pattern="*.json",
		    full.names=TRUE)
df_tmp <- list()
count <- 1
for (f in files)
{
	tmp <- jsonlite::fromJSON(f,flatten=TRUE)
	df_tmp[[count]] <- as.data.frame(tmp)[grepl(columns, colnames(tmp))]
	count <- count + 1
}

# Merge data from multiple .json files
df_merged <- rbind_pages(df_tmp)
df_merged <- dplyr::rename(df_merged, Matrix=problem.name)

# Pre-processing of the input data by splitting it for time and
# storage, and then merging back together
dfTime <- melt(df_merged[,c("Matrix", "optimal.spmv",
			    "spmv.csr.time", "spmv.coo.time",
			    "spmv.ell.time", "spmv.hybrid.time",
			    "spmv.sellp.time")],
	       id.vars=c("Matrix", "optimal.spmv"),
	       variable.name="Format", value.name="Duration")
dfStorage <- melt(df_merged[,c("Matrix",
			       "spmv.csr.storage", "spmv.coo.storage",
			       "spmv.ell.storage", "spmv.hybrid.storage",
			       "spmv.sellp.storage")],
		  id.vars=c("Matrix"),
		  variable.name="Format", value.name="Storage")
dfTime$Format <- gsub('.time', '', dfTime$Format)
dfStorage$Format <- gsub('.storage', '', dfStorage$Format)
df <- merge(dfTime, dfStorage, by=c("Matrix", "Format"))
df$Format <- gsub('spmv.', '', df$Format)

# Additional summary columns
df$OptimalTime <- ifelse(df$optimal.spmv == df$Format, TRUE, FALSE)
dfTmp <- ddply(df, c("Matrix"), summarize, MinSt=min(Storage))
df <- merge(df, dfTmp)
df$MinimalStorage <- ifelse(df$Storage == df$MinSt, TRUE, FALSE)

# Plotting and saving figures
ggplot(df, aes(Format, Duration, fill=OptimalTime)) +
       geom_bar(stat="identity") +
       facet_wrap(~Matrix, scales="free") +
       labs(title="Optimal execution time for SpMVs of different matrices",
	    x="Format", y="Duration [s]") +
       theme_bw() +
       theme(legend.position="none", plot.title=element_text(hjust=0.5))
ggsave(paste(output,"time.pdf", sep="_"),
       width=12, height=6, units="in")
ggplot(df, aes(Format, Storage, fill=MinimalStorage)) +
       geom_bar(stat="identity") +
       facet_wrap(~Matrix, scales="free") +
       labs(title="Minimal storage for SpMVs of different matrices",
	    x="Format", y="Storage [B]") +
       theme_bw() +
       theme(legend.position="none", plot.title=element_text(hjust=0.5))
ggsave(paste(output,"storage.pdf", sep="_"),
       width=12, height=6, units="in")

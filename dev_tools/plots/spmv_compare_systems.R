# Simple script for comparing best SpMVs of different systems
# Reading input files inspired by the tuning_heatmap.R

# Author: Luka Stanisic
# Email: luka.stanisic@huawei.com
# Affiliation: Huawei Technologies Duesseldorf GmbH
# License: 3-clause BSD

# Example of usage for the test data
# (test data was manually modified and it doesnot represent real scenarios)
# Rscript spmv_compare_systems.R testdata/sys1 testdata/sys2 test

# Loaded necessary R libraries
# Suppressing warnings and messages as in R library loading
# is often quite verbose
suppressMessages(suppressWarnings(library(jsonlite)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(plyr)))

# Specify which columns to read from the input file
columns <- "problem.name|optimal.spmv|spmv.*.time"
column_names <- c("Matrix", "csr", "coo", "ell", "hybrid",
		  "sellp", "Optimal", "System")

# Manage arguments
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2)
{
	 stop("Usage: Rscript spmv_compare_systems.R input_directory_1
	      input_directory_2 ... output_graphics\n
	      Example: Rscript spmv_compare_systems.R testdata/sys1
	      testdata/sys2 test\n", call=FALSE)
}
output <- args[length(args)]

df_merged <- list()
inputs <- 1
while (inputs < length(args))
{
	# Read input data
	files <- list.files(paste(args[inputs]), recursive=TRUE,
			    pattern="*.json", full.names=TRUE)
	df_tmp <- list()
	count <- 1
	for (f in files)
	{
		tmp <- jsonlite::fromJSON(f,flatten=TRUE)
		df_tmp[[count]] <- as.data.frame(tmp)[grepl(columns,
							    colnames(tmp))]
		df_tmp[[count]]$System <- args[inputs]
		count <- count + 1
	}

	# Merge data from multiple .json files for a system
	df_merged[[inputs]] <- rbind_pages(df_tmp)

	inputs <- inputs + 1
}

# Merging data frames from multiple systems
df <- rbind_pages(df_merged)
names(df) <- column_names

# Additional summary columns
df$Duration <- apply(df[,c("csr", "coo", "ell", "hybrid", "sellp")], 1, min)
dfTmp <- ddply(df, c("Matrix"), summarize, Minimal=min(Duration))
df <- merge(df, dfTmp)
df$Fastest <- ifelse(df$Minimal == df$Duration, TRUE, FALSE)

# Plotting and saving figure
ggplot(df, aes(System, Duration, fill=Fastest)) +
       geom_bar(stat="identity") +
       geom_text(aes(label=Optimal), vjust=-0.5) +
       facet_wrap(~Matrix, scales="free") +
       labs(title="Fastest SpMVs for different systems",
	    x="System", y="Duration [s]") +
       theme_bw() +
       theme(legend.position="none", plot.title=element_text(hjust=0.5))
ggsave(paste(output,"pdf", sep="."),
       width=12, height=6, units="in")

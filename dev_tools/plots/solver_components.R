# Simple script for comparing performance of different solvers
# Reading input files inspired by the tuning_heatmap.R

# Author: Luka Stanisic
# Email: luka.stanisic@huawei.com
# Affiliation: Huawei Technologies Duesseldorf GmbH
# License: 3-clause BSD

# Example of usage for the test data
# (test data was manually modified and it doesnot represent real scenarios)
# Rscript solver_components.R testdata/sys1/omp/SuiteSparse/MatGroup1/ test

# Loaded necessary R libraries
# Suppressing warnings and messages as in R library loading
# is often quite verbose
suppressMessages(suppressWarnings(library(jsonlite)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(reshape2)))
suppressMessages(suppressWarnings(library(plyr)))

# Specify which columns to read from the input file
columns <- "problem.name|apply"

# Manage arguments
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2)
{
	 stop("Usage: Rscript solver_components.R input_directory
	      output_graphics_file\n
	      Example: Rscript solver_components.R
	      testdata/sys1/omp/SuiteSparse/MatGroup1/ test\n", call=FALSE)
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

# Pre-processing of the input data for each component
dfComponents <- melt(df_merged[grepl("Matrix|apply.components",
			       colnames(df_merged))],
		     id.vars=c("Matrix"), variable.name="Solver",
		     value.name="Duration")
dfComponents$Components <- dfComponents$Solver
dfComponents$Solver <- gsub('.apply.*', '', dfComponents$Solver)
dfComponents$Components <- gsub('solver.*.apply.components.', '',
				dfComponents$Components)
dfComponents$Solver <- gsub('solver.', '', dfComponents$Solver)
dfTmp <- ddply(dfComponents, c("Matrix", "Solver"), summarize,
	       Top1=max(Duration), Sum=sum(Duration))
dfComponents <- merge(dfComponents, dfTmp)
dfTmp <- ddply(dfComponents[dfComponents$Top1 != dfComponents$Duration,],
		c("Matrix", "Solver"), summarize, Top2=max(Duration))
dfComponents <- merge(dfComponents, dfTmp)
dfComponents$Sum <- dfComponents$Sum - dfComponents$Top1 - dfComponents$Top2
dfComponents$Components <- ifelse((dfComponents$Top1 != dfComponents$Duration)
				& (dfComponents$Top2 != dfComponents$Duration),
				'other', dfComponents$Components)

# Plotting and saving figures
ggplot(dfComponents, aes(Solver, Duration, fill=Components)) +
       geom_bar(stat="identity") +
       facet_wrap(~Matrix, scales="free") +
       theme() +
       labs(title="Top 2 components durations for each solver",
	    x="Solver", y="Duration [s]") +
       theme_bw() +
       theme(legend.position="bottom", plot.title=element_text(hjust=0.5))
ggsave(paste(output,"components.pdf", sep="_"),
       width=12, height=6, units="in")
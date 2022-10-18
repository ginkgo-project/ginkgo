# Simple script for comparing performance of different solvers
# Reading input files inspired by the tuning_heatmap.R

# Author: Luka Stanisic
# Email: luka.stanisic@huawei.com
# Affiliation: Huawei Technologies Duesseldorf GmbH
# License: 3-clause BSD

# Example of usage for the test data
# (test data was manually modified and it doesnot represent real scenarios)
# Rscript solver_summary.R testdata/sys1/omp/SuiteSparse/MatGroup1/ test

# Loaded necessary R libraries
# Suppressing warnings and messages as in R library loading
# is often quite verbose
suppressMessages(suppressWarnings(library(jsonlite)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(reshape2)))
suppressMessages(suppressWarnings(library(plyr)))

# Specify which columns to read from the input file
columns <- "problem.name|apply|residual_norm"

# Manage arguments
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2)
{
	 stop("Usage: Rscript solver_summary.R input_directory
	      output_graphics_file\n
	      Example: Rscript solver_summary.R
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

# Pre-processing of the input data for the summary
dfTime <- melt(df_merged[grepl("Matrix|apply.time", colnames(df_merged))],
	       id.vars=c("Matrix"), variable.name="Solver", value.name="Time")
dfTime$Solver <- gsub('.apply.time', '', dfTime$Solver)
dfIterations <- melt(df_merged[grepl("Matrix|apply.iterations",
			       colnames(df_merged))], 
		     id.vars=c("Matrix"), variable.name="Solver",
		     value.name="Iterations")
dfIterations$Solver <- gsub('.apply.iterations', '', dfIterations$Solver)
dfResNorm <- melt(df_merged[grepl("Matrix|residual_norm", colnames(df_merged))],
		  id.vars=c("Matrix"), variable.name="Solver",
		  value.name="ResNorm")
dfResNorm$Solver <- gsub('.residual_norm', '', dfResNorm$Solver)
dfSummary <- merge(merge(dfTime, dfIterations), dfResNorm)
dfSummary$Solver <- gsub('solver.', '', dfSummary$Solver)
dfTmp <- ddply(dfSummary, c("Matrix"), summarize, Minimal=min(Time))
dfSummary <- merge(dfSummary, dfTmp)
dfSummary$Fastest <- ifelse(dfSummary$Minimal == dfSummary$Time, TRUE, FALSE)

# Plotting and saving figures
ggplot(dfSummary, aes(Solver, Time, fill=Fastest)) +
       geom_bar(stat="identity") +
       geom_text(aes(label=Iterations), vjust=-0.5) +
       facet_wrap(~Matrix, scales="free") +
       labs(title="Time and iterations for different solvers",
	    x="Solver", y="Duration [s]") +
       theme_bw() +
       theme(legend.position="none", plot.title=element_text(hjust=0.5))
ggsave(paste(output,"summary_iterations.pdf", sep="_"),
       width=12, height=6, units="in")
ggplot(dfSummary, aes(Solver, Time, fill=Fastest)) +
       geom_bar(stat="identity") +
       geom_text(aes(label=formatC(ResNorm, digits=2, format="e")), vjust=-0.5) +
       facet_wrap(~Matrix, scales="free") +
       labs(title="Time and residual norms for different solvers",
	    x="Solver", y="Duration [s]") +
       theme_bw() +
       theme(legend.position="none", plot.title=element_text(hjust=0.5))
ggsave(paste(output,"summary_residual_norms.pdf", sep="_"),
       width=12, height=6, units="in")
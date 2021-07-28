#source("~/R_libs/lib_common.r")
#source("~/R_libs/array_ops.r")
#source("~/R_libs/matrix_ops.r")
library(RColorBrewer)
library(gplots)
library("MASS")

f_print_array <- function(array, OUTPUT_FILE, header=NULL)
{
  if (file.exists(OUTPUT_FILE)) { file.remove(OUTPUT_FILE) }
  if (!is.null(names(array))) { cat(file=OUTPUT_FILE, append=TRUE, paste(sprintf("%s\t%s", names(array), array), collapse="\n")) }
  else { cat(file=OUTPUT_FILE, append=TRUE, paste(sprintf("%s", array), collapse="\n")) }
  cat(file=OUTPUT_FILE, append=TRUE, "\n")
}

f_print_matrix <- function(matrix, OUTPUT_FILE, colnames=TRUE, header=NULL, rownames=TRUE, contains_lists=FALSE, APPEND=FALSE)
{
  if (colnames == TRUE & rownames == TRUE) colnames <- NA
  if (!is.null(header)) { cat(file=OUTPUT_FILE, append=APPEND, sprintf("%s\n", header)) }
  append_par <- ((!is.null(header)) | APPEND)
  write.table(matrix, file=OUTPUT_FILE, row.names=rownames, col.names=colnames, sep="\t", quote=FALSE, append=append_par)
}

# the sum of all compartments per gene adds up to 1. We have to remove the Punctate.Nuclear data and rescale the remaining values so they still add up to 1.
faux_rescale_values_wo_compartment <- function(data, compartments, to_remove="Punctate.Nuclear")
{
  data[[to_remove]] <- NULL
  to_keep <- compartments[ compartments != to_remove ]
  rescaled <- t(apply(data[,to_keep], 1, function(x) { factor <- 1/sum(x); x*factor }))
  new_data <- as.data.frame(cbind(data[,! colnames(data) %in% to_keep], rescaled))
  for (this_col in to_keep)
  {
    new_data[[this_col]] <- as.numeric(new_data[[this_col]])
  }
  return (new_data)
}

# input_folder: folder in which single cell data is found. In that folder there should be subfolders with the following structure
# - WT_R%s_Plate%d -> data for WT
# - UBP14_R%s_Plate%d -> data for the UBP14 mutant
# - UBP2_R%s_Plate%d -> data for the UBP2 mutant
# - UBP2UBP14_R%s_Plate%d -> data for the double mutant
# %s corresponds to the replicate to consider (see REPLICATE parameter). %d corresponds to the plate number, info should be available for all plates (see NUM_PLATES parameter)
f_run_pipeline <- function(input_folder,         # folder with the input data
                           output_folder,        # folder where to output the results
                           REPLICATE = 2,        # replicate to analyze
                           NUM_PLATES = 12,      # number of plates with data
                           MIN_CELL_COUNT = 15)  # minimum counts per cell 
{
  #MIN_DATA_POINTS <- 2 # otherwise we cannot perform the t-test

  SINGLE_DATA_COLS <- c("character", rep("numeric", 3), rep("character", 3), rep("numeric", 22)) 
  # in the "per_gene" approach, we should read an input file for each gene. It is better to do a "per_file" approach (more efficient)
  ## calculate:
  ## - t-test score per each gene and compartment by comparing WT and alpha screens
  ## - median localization across wells for each gene and compartment and screen (WT and alpha)
  ## - identify the compartment with predominant localization per gene and screen (WT and alpha)
  t_statistics_UBP14 <- list()
  t_statistics_UBP2 <- list()
  t_statistics_UBP2UBP14 <- list()
  med_loc_WT <- list()
  med_loc_UBP14 <- list()
  med_loc_UBP2 <- list()
  med_loc_UBP2UBP14 <- list()
  predominant_loc_WT <- c()
  predominant_loc_UBP14 <- c()
  predominant_loc_UBP2 <- c()
  predominant_loc_UBP2UBP14 <- c()
  n_cells_WT <- c()
  n_cells_UBP14 <- c()
  n_cells_UBP2 <- c()
  n_cells_UBP2UBP14 <- c()
  no_min <- list()
  duplicated <- c()
  
  for (this_plate in (1:NUM_PLATES))
  {
    this_WT_folder <- sprintf("%s/WT_R%s_Plate%d/", input_folder, REPLICATE, this_plate)
    this_UBP14_folder <- sprintf("%s/UBP14_R%s_Plate%d/", input_folder, REPLICATE, this_plate)
    this_UBP2_folder <- sprintf("%s/UBP2_R%s_Plate%d/", input_folder, REPLICATE, this_plate)
    this_UBP2UBP14_folder <- sprintf("%s/UBP2UBP14_R%s_Plate%d/", input_folder, REPLICATE, this_plate)
    # file names are the same for the WT and mutant screens
    this_files <- list.files(path = this_WT_folder, pattern = "Images_predictions_perCell_r")
    for (this_file in this_files)
    {
      # read single cell data for WT, UBP14 and UBP2 mutants, and the double mutant
      WT_data <- read.table(sprintf("%s/%s", this_WT_folder, this_file), header = TRUE, sep = ",", colClasses = SINGLE_DATA_COLS, quote = "\"")
      UBP14_data <- read.table(sprintf("%s/%s", this_UBP14_folder, this_file), header = TRUE, sep = ",", colClasses = SINGLE_DATA_COLS, quote = "\"")
      UBP2_data <- read.table(sprintf("%s/%s", this_UBP2_folder, this_file), header = TRUE, sep = ",", colClasses = SINGLE_DATA_COLS, quote = "\"")
      UBP2UBP14_data <- read.table(sprintf("%s/%s", this_UBP2UBP14_folder, this_file), header = TRUE, sep = ",", colClasses = SINGLE_DATA_COLS, quote = "\"")
  
      # remove Punctate Nuclear and scale the remaining values to add up to 1
      compartments <- colnames(WT_data)[8:ncol(WT_data)]
      WT_data <- faux_rescale_values_wo_compartment(WT_data, compartments)
      UBP14_data <- faux_rescale_values_wo_compartment(UBP14_data, compartments)
      UBP2_data <- faux_rescale_values_wo_compartment(UBP2_data, compartments)
      UBP2UBP14_data <- faux_rescale_values_wo_compartment(UBP2UBP14_data, compartments)
      compartments <- colnames(WT_data)[8:ncol(WT_data)]
  
      this_genes <- unique(WT_data$ORF)
      # keep only genes screened in one plate (also remove "BLANK" and empty ORFs)
      this_genes <- this_genes[ ! this_genes %in% c("BLANK", "") ]
      this_duplicated <- this_genes[ this_genes %in% names(t_statistics_UBP14) ]
      if (length(this_duplicated) > 0)
      {
        this_genes <- this_genes[! this_genes %in% this_duplicated ]
        duplicated <- c(duplicated, this_duplicated)
        cat(sprintf("\n\nGENE %s HAS ALREADY BEEN PROCESSED!!!\nPlate=%s; File=%s\n", paste(this_duplicated, collapse=","), this_plate, this_file))
      }
      for (this_gene in this_genes)
      {
        gene_WT_data <- WT_data[ WT_data$ORF == this_gene, compartments]
        gene_UBP14_data <- UBP14_data[ UBP14_data$ORF == this_gene, compartments]
        gene_UBP2_data <- UBP2_data[ UBP2_data$ORF == this_gene, compartments]
        gene_UBP2UBP14_data <- UBP2UBP14_data[ UBP2UBP14_data$ORF == this_gene, compartments]
        if ((nrow(gene_WT_data) >= MIN_CELL_COUNT) & 
            (nrow(gene_UBP14_data) >= MIN_CELL_COUNT) & 
            (nrow(gene_UBP2_data) >= MIN_CELL_COUNT) &
            (nrow(gene_UBP2UBP14_data) >= MIN_CELL_COUNT) )
        {
          # WT
          med_loc_WT[[this_gene]] <- apply(gene_WT_data, 2, mean, na.rm=TRUE)
          predominant_loc_WT[[this_gene]] <- names(med_loc_WT[[this_gene]])[ order(med_loc_WT[[this_gene]], decreasing = TRUE)[[1]] ]
          n_cells_WT[[this_gene]] <- nrow(gene_WT_data)
          # UBP14
          med_loc_UBP14[[this_gene]] <- apply(gene_UBP14_data, 2, mean, na.rm=TRUE)
          predominant_loc_UBP14[[this_gene]] <- names(med_loc_UBP14[[this_gene]])[ order(med_loc_UBP14[[this_gene]], decreasing = TRUE)[[1]] ]
          n_cells_UBP14[[this_gene]] <- nrow(gene_UBP14_data)
          # UBP2
          med_loc_UBP2[[this_gene]] <- apply(gene_UBP2_data, 2, mean, na.rm=TRUE)
          predominant_loc_UBP2[[this_gene]] <- names(med_loc_UBP2[[this_gene]])[ order(med_loc_UBP2[[this_gene]], decreasing = TRUE)[[1]] ]
          n_cells_UBP2[[this_gene]] <- nrow(gene_UBP2_data)
          # UBP2UBP14
          med_loc_UBP2UBP14[[this_gene]] <- apply(gene_UBP2UBP14_data, 2, mean, na.rm=TRUE)
          predominant_loc_UBP2UBP14[[this_gene]] <- names(med_loc_UBP2UBP14[[this_gene]])[ order(med_loc_UBP2UBP14[[this_gene]], decreasing = TRUE)[[1]] ]
          n_cells_UBP2UBP14[[this_gene]] <- nrow(gene_UBP2UBP14_data)
          # t-statistics
          t_statistics_UBP2[[this_gene]] <- sapply(compartments, function(this_comp)
                                            {
                                              d1 <- gene_WT_data[,this_comp]
                                              d2 <- gene_UBP2_data[,this_comp]
                                              ifelse((length(unique(d1)) > 1) & (length(unique(d2)) > 1), (t.test(d1, d2))$statistic[[1]], NA)
                                            })
          t_statistics_UBP14[[this_gene]] <- sapply(compartments, function(this_comp)
                                            {
                                              d1 <- gene_WT_data[,this_comp]
                                              d2 <- gene_UBP14_data[,this_comp]
                                              ifelse((length(unique(d1)) > 1) & (length(unique(d2)) > 1), (t.test(d1, d2))$statistic[[1]], NA)
                                            })
          t_statistics_UBP2UBP14[[this_gene]] <- sapply(compartments, function(this_comp)
                                              {
                                                d1 <- gene_WT_data[,this_comp]
                                                d2 <- gene_UBP2UBP14_data[,this_comp]
                                                ifelse((length(unique(d1)) > 1) & (length(unique(d2)) > 1), (t.test(d1, d2))$statistic[[1]], NA)
                                              })
          #cat(sprintf("Done %s\n", this_gene))
        }else
        {
          sprintf("%s NO MIN_CELL_COUNT\n", this_gene)
          no_min[[this_gene]] <- c(this_gene, nrow(gene_WT_data), nrow(gene_UBP14_data), nrow(gene_UBP2_data), nrow(gene_UBP2UBP14_data))
        }
      } # for (this_gene in this_genes)
    } # for (this_file in this_files)
    cat(sprintf("Plate %d/%d done (%s)\n", this_plate, NUM_PLATES, date()))
  } # for (this_plate in (1:NUM_PLATES))
  
  # create output folder
  dir.create(output_folder, showWarnings = F, recursive = T)
  no_min_output <- do.call("rbind", no_min)
  colnames(no_min_output) <- c("gene", "n_WT", "n_UBP14", "n_UBP2", "n_UBP2UBP14")
  f_print_matrix(no_min_output, OUTPUT_FILE = sprintf("%s/genes_no_min_data.REPLICATE_%s.csv", output_folder, REPLICATE), rownames = F, colnames = T)
  f_print_array(duplicated, OUTPUT_FILE = sprintf("%s/duplicated_genes.ignored_second.REPLICATE_%s.csv", output_folder, REPLICATE))
  
  all_t_tests_UBP14 <- do.call("rbind", t_statistics_UBP14)
  all_t_tests_UBP2 <- do.call("rbind", t_statistics_UBP2)
  all_t_tests_UBP2UBP14 <- do.call("rbind", t_statistics_UBP2UBP14)
  to_save <- list("UBP14" = all_t_tests_UBP14,
                  "UBP2" = all_t_tests_UBP2,
                  "UBP2UBP14" = all_t_tests_UBP2UBP14)
  save(to_save, file=sprintf("%s/all_t_tests.REPLICATE_%s.R_object", output_folder, REPLICATE))
  
  ## negative t.test values --> mutants have higher localization scores
  ### call mixture model to identify significant changes
  source("lib_mixture_model.r")
  changes_UBP14 <- f_get_significant_changes(all_t_tests_UBP14)
  changes_UBP2 <- f_get_significant_changes(all_t_tests_UBP2)
  changes_UBP2UBP14 <- f_get_significant_changes(all_t_tests_UBP2UBP14)
  
  f_print_matrix(changes_UBP14$df, OUTPUT_FILE = sprintf("%s/UBP14_changes.per_gene.REPLICATE_%s.csv", output_folder, REPLICATE), colnames = T)
  f_print_matrix(changes_UBP14$df_summary, OUTPUT_FILE = sprintf("%s/UBP14_changes.per_compartment.REPLICATE_%s.csv", output_folder, REPLICATE), colnames = T)
  f_print_matrix(changes_UBP2$df, OUTPUT_FILE = sprintf("%s/UBP2_changes.per_gene.REPLICATE_%s.csv", output_folder, REPLICATE), colnames = T)
  f_print_matrix(changes_UBP2$df_summary, OUTPUT_FILE = sprintf("%s/UBP2_changes.per_compartment.REPLICATE_%s.csv", output_folder, REPLICATE), colnames = T)
  f_print_matrix(changes_UBP2UBP14$df, OUTPUT_FILE = sprintf("%s/UBP2UBP14_changes.per_gene.REPLICATE_%s.csv", output_folder, REPLICATE), colnames = T)
  f_print_matrix(changes_UBP2UBP14$df_summary, OUTPUT_FILE = sprintf("%s/UBP2UBP14_changes.per_compartment.REPLICATE_%s.csv", output_folder, REPLICATE), colnames = T)
}


# example
# f_run_pipeline(input_folder="singlecell/", output_folder="output_rep2", REPLICATE = 2)
# f_run_pipeline(input_folder="singlecell/", output_folder="output_rep1", REPLICATE = 1)

library(Seurat)
library(Matrix)
library(data.table)
library(ggplot2)
library(dplyr)

setwd("/Volumes/HD_rafael/DPhil/xenium_lung/inputs")
xenium_data <- readRDS("GSE250346_Seurat_GSE250346_CORRECTED_SEE_RDS_README_082024.rds")

write.table(rownames(xenium_data), "all_gene_names.tsv", 
              row.names = FALSE, col.names = FALSE, quote = FALSE)

metadata <- xenium_data@meta.data

samples <- unique(metadata$sample)

write.csv(metadata, "metadata_lung.csv")

extract_data_for_sample <- function(xenium_data, sample_id) {
  message(paste0("Processing sample: ", sample_id))
  
  message("Subsampling seurat object...")
  subset_obj <- subset(xenium_data, subset = sample == sample_id)
  print(paste0("Dimensions: ", paste(dim(subset_obj), collapse = " x ")))
  
  # 2. Extract Metadata
  message("Extracting cell types, lineage, and coordinates...")
  meta_df <- data.frame(
    cell_id = rownames(subset_obj@meta.data),
    cell_type = subset_obj@meta.data$final_CT,
    lineage = subset_obj@meta.data$final_lineage,
    disease_status = subset_obj@meta.data$disease_status,
    sample_affect = subset_obj@meta.data$sample_affect,
    x_global = subset_obj@meta.data$x_centroid,
    y_global = subset_obj@meta.data$y_centroid
  )
  
  message("Extracting raw counts...")
  counts_mat <- GetAssayData(subset_obj, assay = "RNA", layer = "counts")
  
  # Convert to sparse matrix if it is dense
  if (!inherits(counts_mat, "sparseMatrix")) {
    message("Coercing dense matrix to sparse matrix for writeMM...")
    counts_mat <- as(counts_mat, "sparseMatrix")
  }
  
  message("Creating folder and saving to disk...")
  path_folder <- paste0(sample_id)
  dir.create(path_folder, showWarnings = FALSE, recursive = T)
  
  write.csv(meta_df, file.path(path_folder, "metadata.csv"), row.names = FALSE)
  
  writeMM(counts_mat, file.path(path_folder, "counts.mtx"))

  write.table(rownames(counts_mat), file.path(path_folder, "genes.tsv"), 
              row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(colnames(counts_mat), file.path(path_folder, "barcodes.tsv"), 
              row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  message("Export complete!")
}

for (sample_id in samples) {
  print(sample_id)
  extract_data_for_sample(xenium_data, sample_id)
}
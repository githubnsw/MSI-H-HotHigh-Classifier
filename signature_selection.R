# ==============================================================================
# signature_selection.R
# 20-Gene Signature Selection using GSEA Leading Edge and Hub Genes
# Pipeline: (DEG_sig ∩ GSEA_core) ∪ Hub_genes → Top 20 by |logFC|
# ==============================================================================

library(dplyr)
library(readr)
library(fgsea)
library(msigdbr)

# ==============================================================================
# 1. Load DEG Results
# ==============================================================================

deg_all <- read.csv("results/DEG_HotHigh_vs_rest.csv")

cat("Total DEGs:", nrow(deg_all), "\n")

# ==============================================================================
# 2. Create Ranked Gene List for fgsea (ranked by logFC)
# ==============================================================================

ranked_genes <- deg_all %>%
  dplyr::select(Gene, logFC) %>%
  distinct() %>%
  tibble::deframe()

# Sort descending (required for fgsea)
ranked_genes <- sort(ranked_genes, decreasing = TRUE)

# ==============================================================================
# 3. Load Hallmark Gene Sets
# ==============================================================================

hallmark_sets <- msigdbr(
  species  = "Homo sapiens",
  category = "H"
) %>%
  dplyr::select(gs_name, gene_symbol) %>%
  split(x = .$gene_symbol, f = .$gs_name)

# ==============================================================================
# 4. Run fgsea
# ==============================================================================

set.seed(123)
fgseaRes_HotHigh <- fgsea(
  pathways = hallmark_sets,
  stats    = ranked_genes,
  nperm    = 10000,
  minSize  = 15,
  maxSize  = 500
)

# ==============================================================================
# 5. Extract Leading Edge Genes from Immune Pathways
# ==============================================================================

immune_pathways <- c(
  "HALLMARK_INTERFERON_GAMMA_RESPONSE",
  "HALLMARK_INTERFERON_ALPHA_RESPONSE",
  "HALLMARK_INFLAMMATORY_RESPONSE",
  "HALLMARK_ALLOGRAFT_REJECTION"
)

cat("\n=== Immune Pathway Enrichment Results ===\n")
fgseaRes_HotHigh %>%
  filter(pathway %in% immune_pathways) %>%
  arrange(padj) %>%
  print()

# Check if pathways are present
present_pw <- intersect(immune_pathways, fgseaRes_HotHigh$pathway)
if (length(present_pw) == 0) {
  stop("No immune hallmark pathways found in fgsea results.")
}

# Extract leading edge (core) genes
gsea_core <- unique(unlist(
  fgseaRes_HotHigh$leadingEdge[fgseaRes_HotHigh$pathway %in% present_pw]
))

cat("\nGSEA leading edge genes:", length(gsea_core), "\n")

# ==============================================================================
# 6. Filter Significant DEGs
# ==============================================================================

deg_sig <- deg_all %>%
  filter(adj.P.Val < 0.05, abs(logFC) > 1)

cat("Significant DEGs (adj.P < 0.05, |logFC| > 1):", nrow(deg_sig), "\n")

# ==============================================================================
# 7. Hub Genes (from PPI network analysis - DEG ∩ RF → MCC Top 10)
# ==============================================================================

hub_genes <- c("LCP2", "CD80", "CD74", "IL10RA", "IRF1",
               "CD38", "KLRK1", "CD8A", "CTLA4", "CCR5")

hub_overlap <- intersect(deg_sig$Gene, hub_genes)
cat("Hub genes overlapping with DEG_sig:", length(hub_overlap), "\n")

# ==============================================================================
# 8. Create Candidate Pool: (DEG_sig ∩ GSEA_core) ∪ Hub_overlap
# ==============================================================================

candidate_genes <- intersect(deg_sig$Gene, gsea_core)
candidate_genes <- unique(c(candidate_genes, hub_overlap))

cat("\nCandidate pool size:", length(candidate_genes), "\n")

if (length(candidate_genes) < 20) {
  warning(paste0(
    "Candidate genes < 20 (n=", length(candidate_genes),
    "). Consider relaxing filter criteria."
  ))
}

# ==============================================================================
# 9. Select Top 20 by |logFC|
# ==============================================================================

top20_signature <- deg_sig %>%
  filter(Gene %in% candidate_genes) %>%
  arrange(desc(abs(logFC))) %>%
  slice_head(n = 20) %>%
  pull(Gene)

cat("\n=== Final 20-Gene Signature ===\n")
print(top20_signature)

# ==============================================================================
# 10. Save Results
# ==============================================================================

save(fgseaRes_HotHigh, file = "results/fgseaRes_HotHigh.RData")

write.csv(
  data.frame(Gene = top20_signature),
  "results/20_gene_signature.csv",
  row.names = FALSE
)

# Save with logFC values
signature_with_stats <- deg_sig %>%
  filter(Gene %in% top20_signature) %>%
  arrange(desc(abs(logFC))) %>%
  select(Gene, logFC, adj.P.Val)

write.csv(signature_with_stats, "results/20_gene_signature_with_stats.csv", row.names = FALSE)

cat("\n✅ Saved: results/fgseaRes_HotHigh.RData\n")
cat("✅ Saved: results/20_gene_signature.csv\n")
cat("✅ Saved: results/20_gene_signature_with_stats.csv\n")

#!/usr/bin/env Rscript
# Rendering only: completed JSON reports -> R/grid -> editable TikZ.
# All dimensions below are millimetres at the final 180 mm journal width.
suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(jsonlite)
  library(tikzDevice)
})
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
ROOT <- normalizePath(file.path(dirname(script), "../.."))
OUT <- file.path(ROOT, "manuscripts/figures")
TEX <- file.path(OUT, "tex")
DATA <- file.path(OUT, "source_data")
BUILD <- file.path(ROOT, "outputs/figure_r_tikz")
for (d in c(TEX, DATA, BUILD)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
requested <- commandArgs(trailingOnly = TRUE)

NAVY <- "#234E70"; BLUE <- "#4C78A8"; TEAL <- "#1F8A8A"
GREEN <- "#2E7D32"; ORANGE <- "#C46A1B"; PURPLE <- "#6B4C9A"
RED <- "#B44B4B"; GRAY <- "#73777B"; DARK <- "#263238"
ARMS <- c("Healer" = GREEN, "Non-healer" = ORANGE, "Healthy" = PURPLE)
font_commands <- c("\\usepackage{fontspec}", "\\setmainfont{Arial}", "\\setsansfont{Arial}")
options(tikzDefaultEngine = "xetex",
        tikzUnicodeMetricPackages = c(font_commands, "\\usetikzlibrary{calc}"),
        tikzMetricPackages = c(font_commands, "\\usetikzlibrary{calc}"),
        tikzMetricsDictionary = file.path(BUILD, "arial-metrics"))

theme_set(theme_classic(base_size = 8, base_family = "Arial") + theme(
  # Figure lettering is deliberately monochrome and bold. Data marks and
  # reference lines retain colour; every textual element remains black.
  text = element_text(colour = "black", face = "bold"),
  axis.text = element_text(size = 7.5, colour = "black", face = "bold"),
  axis.title.x = element_text(size = 8, colour = "black", face = "bold", margin = margin(t = 1.4, unit = "mm")),
  axis.title.y = element_text(size = 8, colour = "black", face = "bold", margin = margin(r = 1.3, unit = "mm")),
  axis.ticks = element_line(linewidth = 0.25, colour = GRAY),
  axis.ticks.length = unit(1, "mm"),
  axis.line = element_line(linewidth = 0.25, colour = GRAY),
  panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E2E7EB"),
  panel.grid.minor = element_blank(),
  plot.margin = margin(t = 1.8, r = 1.2, b = 1.2, l = 1.2, unit = "mm"),
  plot.background = element_rect(fill = "white", colour = NA),
  legend.position = "bottom",
  legend.text = element_text(size = 7, colour = "black", face = "bold"),
  legend.title = element_blank(),
  legend.key.size = unit(3.2, "mm"), legend.spacing.x = unit(1.2, "mm"),
  legend.margin = margin(0, 0, 0, 0, unit = "mm"),
  legend.box.spacing = unit(1.3, "mm")
))
sources <- list(); active_sources <- character(); figures <- list()
read_report <- function(path) {
  full <- file.path(ROOT, path)
  d <- fromJSON(full, simplifyVector = FALSE)
  if (identical(d$status, "failed")) stop("Failed source: ", path)
  sources[[path]] <<- digest::digest(file = full, algo = "sha256")
  active_sources <<- unique(c(active_sources, path))
  d
}
repaired <- function(name) read_report(paste0("outputs/analysis/", name, "/report.json"))
historical <- function(name) read_report(paste0("outputs/", name, "/report.json"))
num <- function(rows, key) vapply(rows, function(x) as.numeric(x[[key]]), numeric(1))
chr <- function(rows, key) vapply(rows, function(x) as.character(x[[key]]), character(1))
vector <- function(x) as.numeric(unlist(x, use.names = FALSE))
ordered_factor <- function(x) factor(x, levels = unique(x))
panel <- function(plot, title, data) list(plot = plot, title = title, data = data)
zero_h <- function() geom_hline(yintercept = 0, colour = GRAY, linewidth = .25)
zero_v <- function(x = 0) geom_vline(xintercept = x, colour = GRAY, linewidth = .3, linetype = "dashed")
xgrid <- function() theme(panel.grid.major.y = element_blank(),
                         panel.grid.major.x = element_line(linewidth = .18, colour = "#E2E7EB"))
columns <- function(d, ylabel, colors, limits = NULL) {
  d$label <- ordered_factor(d$label)
  p <- ggplot(d, aes(label, value, fill = label)) +
    geom_col(width = .58) + scale_fill_manual(values = colors) +
    labs(x = NULL, y = ylabel) + theme(legend.position = "none")
  if (!is.null(limits)) p <- p + coord_cartesian(ylim = limits)
  p
}
forest <- function(d, xlabel = "AUC", ref = .5, limits = c(0, 1)) {
  d$label <- factor(d$label, levels = rev(unique(d$label)))
  ggplot(d, aes(value, label)) + zero_v(ref) +
    geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = .15,
                  colour = NAVY, linewidth = .55) +
    geom_point(size = 2, colour = NAVY) +
    scale_x_continuous(limits = limits, expand = expansion(mult = .025)) +
    labs(x = xlabel, y = NULL) + xgrid()
}
record_figure <- function(name, width, height, panels, heading_centres) {
  data_files <- character()
  for (i in seq_along(panels)) {
    path <- file.path(DATA, paste0(name, "_", LETTERS[i], ".csv"))
    write.csv(panels[[i]]$data, path, row.names = FALSE, na = "")
    data_files <- c(data_files, sub(paste0(ROOT, "/"), "", path, fixed = TRUE))
  }
  figures[[name]] <<- list(width_mm = width, height_mm = height,
      panels = vapply(panels, `[[`, character(1), "title"),
      headings = paste(LETTERS[seq_along(panels)], vapply(panels, `[[`, character(1), "title")),
      heading_centres_mm = heading_centres,
      sources = active_sources, data = data_files)
  active_sources <<- character()
  message("TikZ: ", name)
}
draw_figure <- function(name, panels, height = 82, widths = NULL, nrow = 1, gap = 5) {
  ncol <- length(panels) / nrow
  stopifnot(ncol == as.integer(ncol))
  width <- 180; left <- 3; top <- 2; bottom <- 1.5; row_gap <- 4
  if (is.null(widths)) widths <- rep((width - 2 * left - gap * (ncol - 1)) / ncol, ncol)
  stopifnot(abs(sum(widths) + gap * (ncol - 1) - 174) < .01)
  row_h <- (height - top - bottom - row_gap * (nrow - 1)) / nrow
  tikz(file.path(TEX, paste0(name, ".tex")), width = width / 25.4, height = height / 25.4,
       pointsize = 8, standAlone = TRUE, engine = "xetex", sanitize = TRUE,
       timestamp = FALSE, documentDeclaration = "\\documentclass[10pt]{standalone}",
       packages = c("\\usepackage{tikz}", font_commands))
  on.exit(dev.off())
  grid.newpage()
  for (i in seq_along(panels)) {
    col <- (i - 1) %% ncol + 1; row <- (i - 1) %/% ncol + 1
    x <- left + sum(head(widths, col - 1)) + gap * (col - 1)
    y <- height - top - (row - 1) * (row_h + row_gap)
    # One centred text object keeps the panel letter attached to its title.
    grid.text(paste(LETTERS[i], panels[[i]]$title),
              unit(x + widths[col] / 2, "mm"), unit(y, "mm"), just = c("centre", "top"),
              gp = gpar(fontfamily = "Arial", fontsize = 9, fontface = "bold", col = "black"))
    pushViewport(viewport(x = unit(x, "mm"), y = unit(y - 6.3, "mm"),
                          width = unit(widths[col], "mm"), height = unit(row_h - 6.3, "mm"),
                          just = c("left", "top"), clip = "off"))
    grid.draw(ggplotGrob(panels[[i]]$plot))
    popViewport()
  }
  centres <- left + c(0, head(cumsum(widths), -1)) + gap * (seq_len(ncol) - 1) + widths / 2
  record_figure(name, width, height, panels, rep(centres, nrow))
}
write_workflow <- function(name, height, body, steps, heading_centres) {
  colors <- paste0("\\definecolor{", c("navy", "blue", "teal", "orange", "purple", "green", "gray"),
                   "}{HTML}{", sub("#", "", c(NAVY, BLUE, TEAL, ORANGE, PURPLE, GREEN, GRAY)), "}")
  lines <- c("\\documentclass[10pt]{standalone}", "\\usepackage{tikz}", font_commands,
    "\\usetikzlibrary{arrows.meta,patterns}", colors, "\\begin{document}",
    "\\begin{tikzpicture}[x=1mm,y=1mm,font=\\fontsize{8}{10}\\selectfont\\bfseries,text=black,>=Latex]",
    sprintf("\\path[use as bounding box] (0,0) rectangle (180,%s);", height),
    "\\tikzset{step/.style={draw=#1!40,fill=#1!8,rounded corners=1mm,",
    "minimum height=13mm,text width=21mm,align=center,inner sep=1.5mm,text=black},",
    "link/.style={->,draw=gray,line width=0.45pt}}", body,
    "\\end{tikzpicture}", "\\end{document}")
  writeLines(lines, file.path(TEX, paste0(name, ".tex")))
  record_figure(name, 180, height,
                lapply(steps, function(s) panel(NULL, s, data.frame(step = s))), heading_centres)
}
methods <- c("topic_simplex_theta0", "module_score", "pca", "nmf")
method_names <- c("Fibroblast\ntopic", "Module", "PCA", "NMF")

figure1_workflow <- function() {
  biological_measurement_figure()
}
sample_data <- function(d) data.frame(sample = chr(d, "gsm"),
    arm = factor(c("DFU-healer" = "Healer", "DFU-nonhealer" = "Non-healer", "Non-diabetic" = "Healthy")[chr(d, "arm")], levels = names(ARMS)),
    mean = num(d, "topic0_mean"), weight = num(d, "mixing_weight"))
figure2_mixture_semantics <- function() {
  d <- historical("bimodal_stratification")$cohorts$GSE165816_discovery
  s <- repaired("mixture_semantic_inference")
  a <- sample_data(d$per_sample)
  pa <- ggplot(a, aes(weight, mean, colour = arm, shape = arm)) +
    geom_point(size = 1.9, alpha = .9) + scale_colour_manual(values = ARMS) +
    scale_shape_manual(values = c(16, 17, 15)) +
    scale_x_continuous(breaks = c(0, .4, .8)) +
    labs(x = "High-state fraction", y = "Mean fibroblast-topic loading") +
    guides(colour = guide_legend(nrow = 2, byrow = TRUE), shape = guide_legend(nrow = 2, byrow = TRUE))
  b <- data.frame(label = c("Pure\nlow", "Mixed", "Other", "Pure\nhigh"),
      value = vector(d$hypotheses$shape_counts[c("pure_low", "mixed", "other", "pure_high")]))
  pb <- columns(b, "Samples", c("#BCC7CF", BLUE, GRAY, PURPLE), c(0, 18)) +
    scale_y_continuous(breaks = c(0, 5, 10, 15))
  c <- data.frame(label = c("Bootstrap", "Leave-one-out"),
    value = c(s$observed$spearman_rho, NA),
    low = c(s$bootstrap$ci95[[1]], s$leave_one_sample_out$rho_min),
    high = c(s$bootstrap$ci95[[2]], s$leave_one_sample_out$rho_max))
  c$label <- factor(c$label, levels = rev(c$label))
  pc <- ggplot(c, aes(y = label)) +
    geom_segment(aes(x = low, xend = high, yend = label, colour = label), linewidth = 1.2) +
    geom_point(data = c[1, ], aes(x = value), size = 2.2, colour = NAVY) +
    scale_colour_manual(values = c("Bootstrap" = NAVY, "Leave-one-out" = ORANGE)) +
    scale_x_continuous(limits = c(.74, 1), breaks = c(.8, .9, 1)) +
    labs(x = "Spearman ρ", y = NULL) + xgrid() + theme(legend.position = "none")
  draw_figure("figure2_mixture_semantics", list(panel(pa, "Mixture coordinate", a),
    panel(pb, "Sample shape", b), panel(pc, "Uncertainty", c)), height = 70, widths = c(57, 47, 60))
}
figure3_artifact_controls <- function() {
  t <- historical("artifact_triage"); m <- historical("ambient_invariance")
  a <- data.frame(label = c("Stress\nSMD", "PTPRC\nratio"),
     raw = c(t$H1_dissociation$smd, t$H2_doublet$rate_ratio),
     bound = c(t$thresholds$smd_reject, t$thresholds$ptprc_ratio_reject))
  a$value <- a$raw / a$bound
  pa <- columns(a, "Observed / rejection bound", c(ORANGE, ORANGE), c(0, 1.15)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = RED, linewidth = .35)
  b <- data.frame(before = num(t$depth_control_per_sample, "before"), after = num(t$depth_control_per_sample, "after"))
  pb <- ggplot(b, aes(before, after)) + geom_abline(slope = 1, intercept = 0, colour = GRAY, linetype = "dashed", linewidth = .3) +
    geom_point(size = 1.7, colour = BLUE) + coord_fixed(xlim = c(0, .9), ylim = c(0, .9)) +
    scale_x_continuous(breaks = c(0, .4, .8)) + scale_y_continuous(breaks = c(0, .4, .8)) +
    labs(x = "Before matching", y = "After matching")
  c <- data.frame(label = c("All", "RNA-\nexcluded"),
    value = c(m$test_A_purity$bimodal_all$delta_bic, m$test_A_purity$bimodal_pure$delta_bic))
  pc <- columns(c, "BIC (1 component − 2)", c(TEAL, TEAL), c(0, 65))
  draw_figure("figure3_artifact_controls", list(panel(pa, "Technical controls", a),
    panel(pb, "Depth matching", b), panel(pc, "Ambient RNA", c)), height = 66)
}
figure4_outcome_counterexample <- function() {
  a <- sample_data(historical("bimodal_stratification")$cohorts$GSE165816_discovery$per_sample)
  p <- historical("cohort_metadata/gse165816_patient_unit_remap")
  s <- repaired("patient_mapping_equivalence")
  pow <- historical("negative_result_strength")$observed_design_power
  design <- repaired("design_power_simulation")
  pa <- ggplot(a, aes(arm, weight, colour = arm, shape = arm)) +
    geom_point(position = position_jitter(width = .09, height = 0, seed = 4), size = 1.9) +
    stat_summary(fun.min = mean, fun.max = mean, geom = "errorbar", width = .22,
                 linewidth = .65, show.legend = FALSE) +
    scale_colour_manual(values = ARMS) + scale_shape_manual(values = c(16, 17, 15)) +
    scale_x_discrete(labels = c("Healer", "Non-\nhealer", "Healthy")) +
    coord_cartesian(ylim = c(0, .95)) + labs(x = NULL, y = "High-state fraction") + theme(legend.position = "none")
  e <- list(s$effect, p$effect_cell_weighted)
  b <- data.frame(label = c("14 samples", "11 patients"),
    value = num(e, "healer_minus_nonhealer_mean_difference"),
    low = vapply(e, function(x) x$bootstrap_95_ci[[1]], 0),
    high = vapply(e, function(x) x$bootstrap_95_ci[[2]], 0))
  pb <- forest(b, "Healer − non-healer", ref = 0, limits = c(-.01, .09)) +
    scale_x_continuous(limits = c(-.01, .09), breaks = c(0, .04, .08), expand = expansion(mult = .02))
  c <- data.frame(label = c("Observed", "Target"), value = c(pow$power_at_observed, design$protocol$target_power))
  pc <- columns(c, "Power", c(ORANGE, NAVY), c(0, 1)) +
    scale_y_continuous(breaks = c(0, .4, .8), labels = scales::label_percent())
  draw_figure("figure4_outcome_counterexample", list(panel(pa, "Mixture by arm", a),
    panel(pb, "Unit correction", b), panel(pc, "Design power", c)), height = 68, widths = c(54, 63, 47))
}
figure6_sensitivity_negative_controls <- function() {
  s <- historical("method_constant_sensitivity"); n <- historical("pipeline_negative_control")
  a <- data.frame(cut = num(s$cut_sweep$rows, "cut"), rho = num(s$cut_sweep$rows, "A_rho_mean_vs_weight"),
                  p = num(s$cut_sweep$rows, "C_p"))
  line_cut <- function() geom_vline(xintercept = s$cut_sweep$gmm_cut, colour = GRAY, linetype = "dashed", linewidth = .3)
  pa <- ggplot(a, aes(cut, rho)) + geom_line(colour = BLUE, linewidth = .6) + line_cut() +
    coord_cartesian(ylim = c(.88, .97)) + labs(x = "State cut", y = "Spearman ρ")
  pb <- ggplot(a, aes(cut, p)) + geom_hline(yintercept = .05, linetype = "dashed", colour = RED, linewidth = .3) +
    geom_line(colour = ORANGE, linewidth = .55) + geom_point(colour = ORANGE, size = 1.1) + line_cut() +
    coord_cartesian(ylim = c(.035, .07)) + labs(x = "State cut", y = "Healing-arm p")
  k <- data.frame(K = num(s$k_sweep$rows, "K"), rho = num(s$k_sweep$rows, "A_rho_mean_vs_weight"))
  pc <- ggplot(k, aes(K, rho)) + geom_line(colour = BLUE, linewidth = .5) + geom_point(size = 2, colour = c(ORANGE, BLUE, BLUE)) +
    scale_x_continuous(breaks = c(10, 15, 20)) + coord_cartesian(ylim = c(.88, .97)) + labs(x = "Topics K", y = "Spearman ρ")
  d <- data.frame(label = c("Observed", "Null 95%"), value = c(n$N1_label_permutation$observed, NA),
    low = c(NA, n$N1_label_permutation$null_ci95[[1]]), high = c(NA, n$N1_label_permutation$null_ci95[[2]]))
  d$label <- factor(d$label, levels = rev(d$label))
  pd <- ggplot(d, aes(y = label)) + zero_v() +
    geom_segment(data = d[2, ], aes(x = low, xend = high, yend = label), colour = BLUE, linewidth = 1.5) +
    geom_point(data = d[1, ], aes(x = value), colour = RED, size = 2) +
    scale_x_continuous(limits = c(-.4, .4), breaks = c(-.3, 0, .3)) + labs(x = "Arm difference", y = NULL) + xgrid()
  e <- data.frame(label = c("Intact", "Shuffled"), value = c(n$N2_structureless_input$perplexity_real, n$N2_structureless_input$perplexity_null))
  pe <- columns(e, "Cell perplexity", c(TEAL, GRAY), c(0, 16)) + scale_y_continuous(breaks = c(0, 5, 10, 15))
  f <- data.frame(label = c("Observed p", "Null rate"), value = c(n$N3_permuted_positive_control$p_real, n$N3_permuted_positive_control$fraction_significant_under_permutation))
  pf <- columns(f, "Probability", c(RED, BLUE), c(0, .065)) +
    geom_hline(yintercept = .05, linetype = "dashed", colour = GRAY, linewidth = .3) +
    scale_x_discrete(labels = c("Observed\np", "Null\nrate"))
  draw_figure("figure6_sensitivity_negative_controls", list(panel(pa, "Semantic association", a),
    panel(pb, "Healing contrast", a), panel(pc, "Topic count", k), panel(pd, "Label permutation", d),
    panel(pe, "Structureless input", e), panel(pf, "Null calibration", f)), nrow = 2, height = 119)
}
figure7_representation_inference <- function() {
  r <- repaired("representation_inference"); b <- historical("representation_benchmark")
  a <- data.frame(label = method_names, value = num(r$auc[methods], "loso_auc"),
    low = vapply(r$auc[methods], function(x) x$bootstrap_95_ci[[1]], 0),
    high = vapply(r$auc[methods], function(x) x$bootstrap_95_ci[[2]], 0),
    null_mean = num(r$permutation_null[methods], "null_auc_mean"))
  pa <- forest(a) + geom_errorbar(aes(xmin = null_mean, xmax = null_mean),
    orientation = "y", width = .17, linewidth = .55, colour = ORANGE)
  corr <- diag(4)
  for (i in 1:3) for (j in (i + 1):4) corr[i, j] <- corr[j, i] <- b$cross_method_spearman_representative[[paste(methods[i], methods[j], sep = "__")]]
  c <- expand.grid(row = 1:4, col = 1:4); c$rho <- corr[cbind(c$row, c$col)]
  pc <- ggplot(c, aes(col, row, fill = rho)) + geom_tile(colour = "white", linewidth = .35) +
    geom_text(aes(label = sprintf("%.2f", rho)), size = 7.5 / .pt,
              family = "Arial", fontface = "bold", colour = "black") +
    scale_fill_gradient2(low = "#AFC5D6", mid = "#F8FAFB", high = "#E1A1A1", limits = c(-1, 1), name = "ρ") +
    scale_x_continuous(breaks = 1:4, labels = method_names, expand = c(0, 0)) +
    scale_y_reverse(breaks = 1:4, labels = method_names, expand = c(0, 0)) +
    coord_fixed() + labs(x = NULL, y = NULL) +
    theme(axis.line = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank(),
          legend.position = "right", legend.key.height = unit(15, "mm"), legend.key.width = unit(2.4, "mm"),
          legend.title = element_text(size = 8, colour = "black", face = "bold")) + guides(fill = guide_colorbar(display = "rectangles", nbin = 60,
              barheight = unit(28, "mm"), barwidth = unit(2.4, "mm")))
  draw_figure("figure7_representation_inference", list(panel(pa, "Leave-one-sample AUC", a),
    panel(pc, "Score correlation", c)), height = 71, widths = c(81, 88))
}
figure8_patient_unit_anatomy <- function() {
  r <- repaired("patient_unit_representation_benchmark"); t <- repaired("paired_anatomical_control")
  a <- data.frame(label = method_names, value = num(r$auc[methods], "auc"),
    low = vapply(r$auc[methods], function(x) x$bootstrap_95_ci[[1]], 0),
    high = vapply(r$auc[methods], function(x) x$bootstrap_95_ci[[2]], 0))
  c <- t$topic0_contrasts$topic0_all_mean
  b <- data.frame(label = "Fibroblast\ntopic", value = c$observed_difference_foot_minus_forearm,
                  low = c$bootstrap_95_ci[[1]], high = c$bootstrap_95_ci[[2]])
  pb <- ggplot(b, aes(label, value)) + zero_h() + geom_errorbar(aes(ymin = low, ymax = high), width = .15, colour = GREEN, linewidth = .6) +
    geom_point(size = 2.2, colour = GREEN) + coord_cartesian(ylim = c(-.03, .125)) + labs(x = NULL, y = "Foot − forearm loading")
  draw_figure("figure8_patient_unit_anatomy", list(panel(forest(a), "Patient-unit AUC", a),
    panel(pb, "Paired anatomy", b)), height = 66, widths = c(88, 81))
}
figure9_external_construct_audit <- function() {
  i <- repaired("external_immune_axis"); g <- repaired("external_gse268834_projection"); h <- repaired("external_gse248247_projection")
  cohort <- c("GSE223964", "GSE245703", "GSE268834")
  a <- data.frame(cohort = rep(cohort, 2), type = rep(c("Immune fraction", "B/plasma share"), each = 3),
    rho = c(vector(i$cross_cohort$strict_immune_rho[cohort[1:2]]), g$sample_level_associations$immune_fraction_exclusive$rho,
        vapply(i$cohorts[cohort[1:2]], function(x) x$sample_level_associations$b_plasma_share_of_immune$rho, 0),
        g$sample_level_associations$b_plasma_share_of_immune$rho))
  a$cohort <- factor(a$cohort, levels = rev(cohort))
  pa <- ggplot(a, aes(rho, cohort, colour = type, shape = type)) + zero_v() +
    geom_point(size = 2, position = position_dodge(width = .35)) +
    scale_colour_manual(values = c("Immune fraction" = TEAL, "B/plasma share" = PURPLE)) +
    scale_shape_manual(values = c("Immune fraction" = 16, "B/plasma share" = 15)) +
    scale_x_continuous(limits = c(-1, 1), breaks = c(-1, 0, 1)) +
    labs(x = "Spearman ρ", y = NULL) + xgrid() + guides(colour = guide_legend(ncol = 1), shape = guide_legend(ncol = 1))
  b <- data.frame(cohort = factor(cohort, levels = rev(cohort)),
    rho = c(vapply(i$cohorts[cohort[1:2]], function(x) x$sample_level_associations$median_genes_per_cell$rho, 0), g$sample_level_associations$median_genes_per_cell$rho))
  pb <- ggplot(b, aes(rho, cohort)) + zero_v() + geom_point(colour = ORANGE, size = 2) +
    scale_x_continuous(limits = c(-1, 1), breaks = c(-1, 0, 1)) + labs(x = "Spearman ρ", y = NULL) + xgrid()
  c <- data.frame(label = c("GSE268834", "GSE248247"),
    value = c(g$sample_level_group_contrasts$topic0_fibro_mean$difference_diabetic_minus_non_diabetic,
              h$sample_level_group_contrasts$topic0_fibro_mean$difference_diabetic_minus_non_diabetic))
  pc <- ggplot(c, aes(factor(label, levels = rev(label)), value)) + zero_h() +
    geom_point(aes(colour = label), size = 2.1) + scale_colour_manual(values = c("GSE268834" = GREEN, "GSE248247" = GRAY)) +
    scale_x_discrete(labels = c("GSE248247" = "248247", "GSE268834" = "268834")) +
    coord_cartesian(ylim = c(-.04, .075)) + labs(x = "GSE series", y = "Diabetic − non-diabetic") + theme(legend.position = "none")
  draw_figure("figure9_external_construct_audit", list(panel(pa, "Immune context", a),
    panel(pb, "Detected genes", b), panel(pc, "Group contrast", c)), height = 73, widths = c(59, 59, 46))
}


source(file.path(ROOT, "manuscripts/r_tikz/biological_figures.R"))
names_all <- c("figure1_workflow", "figure2_mixture_semantics", "figure3_artifact_controls", "figure4_outcome_counterexample", "figure5_patient_readouts", "figure6_sensitivity_negative_controls", "figure7_representation_inference", "figure8_patient_unit_anatomy", "figure9_external_construct_audit")
selected <- if (length(requested)) requested else names_all
if (!all(selected %in% names_all)) stop("Unknown figure name")
for (name in selected) get(name, mode = "function")()
manifest <- list(renderer = "R + ggplot2/grid + tikzDevice + XeLaTeX", font = "Arial",
  text_colour = "#000000", text_weight = "bold",
  base_font_pt = 8, tick_font_pt = 7.5, label_font_pt = 11, journal_width_mm = 180,
  figures = figures, sources = sources, R = R.version.string,
  packages = lapply(c("ggplot2", "tikzDevice", "jsonlite", "digest"), function(p) list(package = p, version = as.character(packageVersion(p)))),
  design_sources = setNames(lapply(c("construct_design.tikz"),
    function(f) digest::digest(file = file.path(ROOT, "manuscripts/r_tikz", f), algo = "sha256")),
    paste0("manuscripts/r_tikz/", c("construct_design.tikz"))),
  script_sha256 = digest::digest(file = normalizePath(script), algo = "sha256"))
manifest$auxiliary_sources <- setNames(list(digest::digest(file=file.path(ROOT,"manuscripts/r_tikz/biological_figures.R"),algo="sha256")),"manuscripts/r_tikz/biological_figures.R")
write_json(manifest, file.path(BUILD, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(capture.output(sessionInfo()), file.path(BUILD, "sessionInfo.txt"))

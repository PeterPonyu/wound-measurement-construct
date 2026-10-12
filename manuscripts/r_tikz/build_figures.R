#!/usr/bin/env Rscript
# Rendering only: completed JSON reports -> R/grid -> editable TikZ.
# All dimensions below are millimetres at the final 180 mm journal width.
suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(jsonlite)
  library(tikzDevice)
  library(patchwork)
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
DONORS <- c("PWH26" = BLUE, "PWH27" = ORANGE, "PWH28" = TEAL)
font_commands <- c("\\usepackage{fontspec}", "\\setmainfont{Arial}", "\\setsansfont{Arial}")
options(tikzDefaultEngine = "xetex",
        tikzUnicodeMetricPackages = c(font_commands, "\\usetikzlibrary{calc}"),
        tikzMetricPackages = c(font_commands, "\\usetikzlibrary{calc}"),
        tikzMetricsDictionary = file.path(BUILD, "arial-metrics"))

theme_set(theme_classic(base_size = 8, base_family = "Arial") + theme(
  # Only the separate panel letters are bold; ordinary figure text is regular.
  text = element_text(colour = "black", face = "plain"),
  axis.text = element_text(size = 7.5, colour = "black", face = "plain"),
  axis.title.x = element_text(size = 8, colour = "black", face = "plain", margin = margin(t = 1.4, unit = "mm")),
  axis.title.y = element_text(size = 8, colour = "black", face = "plain", margin = margin(r = 1.3, unit = "mm")),
  axis.ticks = element_line(linewidth = 0.25, colour = GRAY),
  axis.ticks.length = unit(1, "mm"),
  axis.line = element_line(linewidth = 0.25, colour = GRAY),
  panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E2E7EB", inherit.blank = TRUE),
  panel.grid.minor = element_blank(),
  plot.margin = margin(t = 1.8, r = 1.2, b = 1.2, l = 1.2, unit = "mm"),
  plot.background = element_rect(fill = "white", colour = NA),
  legend.position = "bottom",
  legend.text = element_text(size = 7, colour = "black", face = "plain"),
  strip.text = element_text(size = 7.5, colour = "black", face = "plain"),
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
read_panel_csv <- function(path) {
  full <- file.path(ROOT, path)
  sources[[path]] <<- digest::digest(file = full, algo = "sha256")
  active_sources <<- unique(c(active_sources, path))
  read.csv(full, stringsAsFactors = FALSE)
}
measurement_reports <- c("patient_mapping_equivalence", "representation_benchmark",
  "representation_inference", "patient_unit_representation_benchmark", "design_power_simulation")
provenance_repair_reports <- c(
  time_parameterisation = "outputs/scientific_revision_20260926/provenance_repair/time_parameterisation/report.json",
  donor_transfer_flow = "outputs/scientific_revision_20260926/provenance_repair/donor_transfer_flow/report.json",
  donor_count_curve = "outputs/scientific_revision_20260926/provenance_repair/donor_count_curve/report.json"
)
repaired <- function(name) {
  if (name %in% names(provenance_repair_reports)) return(read_report(provenance_repair_reports[[name]]))
  prefix <- if (name %in% measurement_reports) "outputs/scientific_revision_20260922/measurement/" else "outputs/analysis/"
  read_report(paste0(prefix, name, "/report.json"))
}
historical <- function(name) {
  if (name == "cohort_metadata/gse165816_patient_unit_remap")
    return(read_report("outputs/scientific_revision_20260922/measurement/patient_unit_remap/report.json"))
  if (name == "representation_benchmark") return(repaired(name))
  read_report(paste0("outputs/", name, "/report.json"))
}
num <- function(rows, key) vapply(rows, function(x) as.numeric(x[[key]]), numeric(1))
chr <- function(rows, key) vapply(rows, function(x) as.character(x[[key]]), character(1))
vector <- function(x) as.numeric(unlist(x, use.names = FALSE))
ordered_factor <- function(x) factor(x, levels = unique(x))
panel <- function(plot, title, data) list(plot = plot, title = title, data = data)
zero_h <- function() geom_hline(yintercept = 0, colour = GRAY, linewidth = .25)
zero_v <- function(x = 0) geom_vline(xintercept = x, colour = GRAY, linewidth = .3, linetype = "dashed")
xgrid <- function() theme(panel.grid.major.y = element_blank(),
                         panel.grid.major.x = element_line(linewidth = .18, colour = "#E2E7EB", inherit.blank = TRUE))
legend_titles <- function() theme(legend.title = element_text(size = 7, colour = "black", face = "plain"))
# Wrap axis titles at a parenthetical or into balanced lines; never split "n = 25".
wrap_axis_title <- function(label, width) {
  if (nchar(label) <= width) return(label)
  protected <- gsub(" = ", "\u00a7=\u00a7", label, fixed = TRUE)
  open <- regexpr(" (", protected, fixed = TRUE)
  if (open > 1 && open - 1 <= width && nchar(protected) - open <= width) {
    lines <- c(substr(protected, 1, open - 1), substr(protected, open + 1, nchar(protected)))
  } else {
    greedy <- strwrap(protected, width = width)
    target <- max(ceiling(nchar(protected) / length(greedy)), 8)
    lines <- strwrap(protected, width = target + 1)
    if (length(lines) > length(greedy) || any(nchar(lines) > width)) lines <- greedy
  }
  gsub("\u00a7", " ", paste(lines, collapse = intToUtf8(10)), fixed = TRUE)
}
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
record_figure <- function(name, width, height, panels, heading_centres, label_lefts, heading_tops, panel_widths = rep(width / max(1, length(panels)), length(panels))) {
  data_files <- character()
  for (i in seq_along(panels)) {
    path <- file.path(DATA, paste0(name, "_", LETTERS[i], ".csv"))
    write.csv(panels[[i]]$data, path, row.names = FALSE, na = "")
    data_files <- c(data_files, sub(paste0(ROOT, "/"), "", path, fixed = TRUE))
  }
  figures[[name]] <<- list(width_mm = width, height_mm = height,
      panels = vapply(panels, `[[`, character(1), "title"),
      headings = vapply(panels, `[[`, character(1), "title"),
      panel_labels = if (length(panels) > 1) LETTERS[seq_along(panels)] else character(),
      label_left_mm = label_lefts, heading_top_mm = heading_tops,
      panel_width_mm = panel_widths, heading_alignment = "panel-centred",
      heading_centres_mm = heading_centres,
      sources = active_sources, data = data_files)
  active_sources <<- character()
  message("TikZ: ", name)
}
draw_figure <- function(name, panels, height = 82, widths = NULL, nrow = 1, gap = 5, layout = NULL) {
  ncol <- length(panels) / nrow
  stopifnot(ncol == as.integer(ncol))
  width <- 180; left <- 3; top <- 2; bottom <- 1.5; row_gap <- 4
  if (is.null(widths)) widths <- rep((width - 2 * left - gap * (ncol - 1)) / ncol, ncol)
  stopifnot(abs(sum(widths) + gap * (ncol - 1) - 174) < .01)
  row_h <- (height - top - bottom - row_gap * (nrow - 1)) / nrow
  if (is.null(layout)) {
    layout <- lapply(seq_along(panels), function(i) {
      col <- (i - 1) %% ncol + 1; row <- (i - 1) %/% ncol + 1
      c(left + sum(head(widths, col - 1)) + gap * (col - 1),
        top + (row - 1) * (row_h + row_gap), widths[col], row_h)
    })
  }
  stopifnot(length(layout) == length(panels))
  tikz(file.path(TEX, paste0(name, ".tex")), width = width / 25.4, height = height / 25.4,
       pointsize = 8, standAlone = TRUE, engine = "xetex", sanitize = TRUE,
       timestamp = FALSE, documentDeclaration = "\\documentclass[10pt]{standalone}",
       packages = c("\\usepackage{tikz,graphicx}", font_commands))
  on.exit(dev.off())
  grid.newpage()
  for (i in seq_along(panels)) {
    rectangle <- layout[[i]]
    x <- rectangle[1]; y <- height - rectangle[2]
    panel_w <- rectangle[3]; panel_h <- rectangle[4]
    # Centre every title on its full panel, independently of the left-hand letter.
    # Single-panel figures have a title only; PDF validation checks both objects.
    labelled <- length(panels) > 1
    if (labelled) grid.text(LETTERS[i], unit(x, "mm"), unit(y, "mm"),
              just = c("left", "top"),
              gp = gpar(fontfamily = "Arial", fontsize = 11, fontface = "bold", col = "black"))
    grid.text(panels[[i]]$title,
              unit(x + panel_w / 2, "mm"),
              unit(y - .5, "mm"), just = c("centre", "top"),
              gp = gpar(fontfamily = "Arial", fontsize = 9, fontface = "plain", col = "black"))
    if (inherits(panels[[i]]$plot, "vector_panel")) {
      # Direct PDF inclusion keeps the hand-maintained tissue artwork out of
      # ggplot scale/clip handling. The same asset is delivered independently.
      tikzAnnotate(sprintf("\\node[anchor=north west,inner sep=0pt] at (%.6f,%.6f) {\\includegraphics[width=%.6fmm]{../%s.pdf}};",
                           x * 72.27 / 25.4, (y - 6.3) * 72.27 / 25.4,
                           panel_w, panels[[i]]$plot$asset))
      next
    }
    pushViewport(viewport(x = unit(x, "mm"), y = unit(y - 6.3, "mm"),
                          width = unit(panel_w, "mm"), height = unit(panel_h - 6.3, "mm"),
                          just = c("left", "top"), clip = "off"))
    plot <- panels[[i]]$plot
    if (inherits(plot, "ggplot") && !inherits(plot, "patchwork")) {
      # Keep full-size lettering inside short panel rows: wrap the wording,
      # not the artwork. A long rotated y title otherwise crosses the header.
      # About 1.55 mm per 8 pt Arial character after tick-label space.
      y_width <- if (panel_h < 55) 18 else 24
      x_width <- floor((panel_w - 14) / 1.55)
      for (axis in c("x", "y")) {
        label <- plot$labels[[axis]]
        if (is.character(label) && length(label) == 1 && !grepl(intToUtf8(10), label, fixed = TRUE)) {
          plot$labels[[axis]] <- wrap_axis_title(label, if (axis == "y") y_width else x_width)
        }
      }
    }
    grob <- if (inherits(plot, "patchwork")) patchworkGrob(plot) else ggplotGrob(plot)
    grid.draw(grob)
    popViewport()
  }
  lefts <- vapply(layout, function(r) r[1], numeric(1))
  tops <- vapply(layout, function(r) r[2], numeric(1))
  panel_widths <- vapply(layout, function(r) r[3], numeric(1))
  centres <- lefts + panel_widths / 2
  record_figure(name, width, height, panels,
                centres, lefts, tops, panel_widths)
}

# Figure 1 gives the physical sampling object the complete opening row.
# B--D retain measured expression context; E--F retain wide numerical panels.
physical_object_layout <- function() list(
  c(3, 2, 174, 60.3),
  c(3, 68.3, 54.6666667, 74), c(62.6666667, 68.3, 54.6666667, 74),
  c(122.3333333, 68.3, 54.6666667, 74),
  c(3, 148.3, 84.5, 62.2), c(92.5, 148.3, 84.5, 62.2))
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
                lapply(steps, function(s) panel(NULL, s, data.frame(step = s))), heading_centres,
                seq(0, by = 180 / max(1, length(steps)), length.out = length(steps)),
                rep(2, length(steps)), rep(180 / max(1, length(steps)), length(steps)))
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
  s <- read_report("outputs/scientific_revision_20260922/measurement/robustness_extensions/report.json")$patient$patient_semantics
  a <- sample_data(d$per_sample)
  pa <- ggplot(a, aes(weight, mean, colour = arm, shape = arm)) +
    geom_point(size = 1.9, alpha = .9) + scale_colour_manual(values = ARMS) +
    scale_shape_manual(values = c(16, 17, 15)) +
    scale_x_continuous(breaks = c(0, .4, .8)) +
    labs(x = "High-state fraction", y = "Mean fibroblast-topic loading") +
    guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
  b <- a
  b$category <- factor(chr(d$per_sample, "shape"),
    levels = c("pure_low", "mixed", "other", "pure_high"),
    labels = c("Pure low", "Mixed", "Other", "Pure high"))
  pb <- ggplot(b, aes(category, fill = arm)) + geom_bar(width = .65) +
    scale_fill_manual(values = ARMS) + labs(x = NULL, y = "Specimens by shape and arm") +
    scale_x_discrete(labels = c("Pure\nlow", "Mixed", "Other", "Pure\nhigh")) +
    scale_y_continuous(breaks = c(0, 5, 10, 15)) + guides(fill = guide_legend(nrow = 1))
  c <- data.frame(label = c("Bootstrap", "Leave-one-out"),
    value = c(s$rho, NA),
    low = c(s$bootstrap_95_interval[[1]], s$omission_range[[1]]),
    high = c(s$bootstrap_95_interval[[2]], s$omission_range[[2]]),
    omitted_patient = NA_character_, kind = "summary")
  omissions <- do.call(rbind, lapply(s$leave_one_patient_out, as.data.frame))
  omitted <- data.frame(label = "Leave-one-out", value = omissions$rho, low = NA_real_, high = NA_real_,
    omitted_patient = omissions$omitted_patient, kind = "patient_omission")
  c$label <- factor(c$label, levels = rev(c$label))
  pc <- ggplot(c, aes(y = label)) +
    geom_segment(aes(x = low, xend = high, yend = label, colour = label), linewidth = 1.2) +
    geom_point(data = c[1, ], aes(x = value), size = 2.2, colour = NAVY) +
    geom_point(data = omitted, aes(x = value),
      position = position_jitter(width = 0, height = .18, seed = 17), size = 1.4, colour = ORANGE, alpha = .8) +
    geom_vline(xintercept = s$rho, colour = GRAY, linetype = "dotted", linewidth = .3) +
    scale_colour_manual(values = c("Bootstrap" = NAVY, "Leave-one-out" = ORANGE)) +
    scale_x_continuous(limits = c(.74, 1), breaks = c(.8, .9, 1)) +
    labs(x = "Spearman ρ (20 omissions shown)", y = NULL) + xgrid() + theme(legend.position = "none")
  c <- rbind(c, omitted)
  quantiles <- data.frame(sample = chr(d$per_sample, "gsm"),
    median = num(d$per_sample, "topic0_p50"), low = num(d$per_sample, "topic0_p10"),
    high = num(d$per_sample, "topic0_p90"), arm = a$arm)
  quantiles <- quantiles[order(quantiles$median), ]
  quantiles$display_order <- seq_len(nrow(quantiles))
  pd <- ggplot(quantiles, aes(display_order, median, colour = arm, shape = arm)) +
    geom_linerange(aes(ymin = low, ymax = high), linewidth = .45) +
    geom_point(size = 1.5) + scale_colour_manual(values = ARMS) +
    scale_shape_manual(values = c(16, 17, 15)) +
    geom_hline(yintercept = .18429584801197052, linetype = "dashed", colour = GRAY, linewidth = .3) +
    labs(x = "Specimens ordered by median (n = 25)", y = "Median and 10th-90th percentiles") +
    guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
  patients <- do.call(rbind, lapply(s$patients, as.data.frame))
  pe <- ggplot(patients, aes(high_state_fraction, fibroblast_mean, shape = factor(specimens))) +
    geom_point(colour = BLUE, size = 1.9) + scale_shape_manual(values = c(16, 17),
      labels = c("One specimen", "Two specimens")) +
    labs(x = "Patient high-state fraction", y = "Patient fibroblast mean") +
    guides(shape = guide_legend(nrow = 1))
  decomposition <- read_panel_csv("outputs/scientific_revision_20260922/measurement/mathematical_audit/state_decomposition.csv")
  within <- rbind(data.frame(specimen = decomposition$specimen, fraction = decomposition$high_state_fraction,
    bin = "Low bin", loading = decomposition$low_state_mean),
    data.frame(specimen = decomposition$specimen, fraction = decomposition$high_state_fraction,
    bin = "High bin", loading = decomposition$high_state_mean))
  pf <- ggplot(within[is.finite(within$loading), ], aes(fraction, loading, colour = bin, shape = bin)) +
    geom_point(size = 1.6) + scale_colour_manual(values = c("Low bin" = GRAY, "High bin" = TEAL)) +
    scale_shape_manual(values = c("Low bin" = 16, "High bin" = 17)) +
    labs(x = "Specimen high-state fraction", y = "Mean loading within each bin") +
    guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
  draw_figure("figure2_mixture_semantics", list(panel(pa, "Mean and fraction", a),
    panel(pb, "Sample shape", b), panel(pc, "Patient uncertainty", c),
    panel(pd, "Within-specimen spread", quantiles), panel(pe, "Patient-level observations", patients),
    panel(pf, "Within-bin variation", within)), height = 158, nrow = 3)
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
  c <- data.frame(label = c("All", "RNA-\nexcluded", "Immune\nresidual", "Placebo\nresidual"),
    value = c(m$test_A_purity$bimodal_all$delta_bic, m$test_A_purity$bimodal_pure$delta_bic,
      m$test_B_residual$bimodal_residual$delta_bic, m$test_B_placebo$bimodal_residual$delta_bic),
    specimens = c(m$test_A_purity$n_samples, m$test_A_purity$n_samples,
      m$test_B_residual$bimodal_residual$n, m$test_B_placebo$bimodal_residual$n))
  pc <- columns(c, "ΔBIC (one − two components)", c(TEAL, TEAL, ORANGE, GRAY), c(-12, 65)) + zero_h()
  depth <- read_report("outputs/scientific_revision_20260925/full_pipeline_depth/full/report.json")
  d <- do.call(rbind, lapply(depth$pooled_qc_gate_by_run, as.data.frame))
  depth_plot <- function(column, ylabel) {
    ggplot(d, aes(100 * rate, 100 * .data[[column]], group = factor(seed))) +
      geom_line(data = d[d$rate < 1, ], aes(colour = factor(seed)), linewidth = .4) +
      geom_point(aes(shape = factor(seed)), size = 1.7, colour = NAVY) +
      scale_x_continuous(breaks = c(10, 25, 50, 75, 100)) +
      labs(x = "Expected count retention (%)", y = ylabel) +
      theme(legend.position = "none")
  }
  pd <- depth_plot("original_qc_retention_fraction", "Baseline QC retained (%)")
  pe <- depth_plot("common_qc_lineage_change_fraction", "Lineage switching (%)")
  f <- do.call(rbind, lapply(depth$full_pipeline_outcome_seed_ranges, as.data.frame))
  f <- f[f$readout != "all_cell_topic0_mean", ]
  f$readout <- factor(f$readout, levels = c("fibroblast_topic0_mean", "fibroblast_high_state_fraction", "fibroblast_fraction"),
    labels = c("Fibroblast mean", "High-state fraction", "Composition"))
  pf <- ggplot(f, aes(100 * rate, group = readout, colour = readout, shape = readout)) + zero_h() +
    geom_errorbar(aes(ymin = difference_min, ymax = difference_max), width = 2, linewidth = .5) +
    geom_point(aes(y = difference_min), size = 1.5) + geom_point(aes(y = difference_max), size = 1.5) +
    scale_colour_manual(values = c(NAVY, TEAL, ORANGE)) +
    scale_x_continuous(breaks = c(10, 25, 50, 75, 100)) +
    labs(x = "Expected count retention (%)", y = "Healed minus not healed") +
    guides(colour = guide_legend(nrow = 2, byrow = TRUE), shape = guide_legend(nrow = 2, byrow = TRUE))
  draw_figure("figure3_artifact_controls", list(panel(pa, "Technical controls", a),
    panel(pb, "Conditional depth match", b), panel(pc, "Exclusion and residuals", c),
    panel(pd, "Whole-pipeline retention", d), panel(pe, "Whole-pipeline gate", d),
    panel(pf, "Outcome sensitivity", f)), height = 154, nrow = 3)
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
  omitted <- read_panel_csv("outputs/scientific_revision_20260922/measurement/robustness_extensions/patient_omission.csv")
  pb <- pb + geom_point(data = omitted, aes(x = healing_minus_nonhealing, y = "11 patients"),
    inherit.aes = FALSE, position = position_jitter(width = 0, height = .16, seed = 17),
    colour = ORANGE, alpha = .7, size = 1.4)
  b <- rbind(transform(b, kind = "estimate", omitted_patient = NA_character_),
    data.frame(label = "11 patients", value = omitted$healing_minus_nonhealing,
      low = NA_real_, high = NA_real_, kind = "patient_omission", omitted_patient = omitted$omitted_patient))
  c <- do.call(rbind, lapply(names(design$design_a_detect_difference), function(key) {
    row <- as.data.frame(design$design_a_detect_difference[[key]])
    row$difference <- as.numeric(sub("difference_", "", key, fixed = TRUE)); row
  }))
  c$target_power <- design$protocol$target_power
  c$historical_fraction_power <- pow$power_at_observed
  pc <- ggplot(c, aes(difference, total)) + geom_line(colour = NAVY, linewidth = .45) +
    geom_point(aes(size = power), colour = NAVY) +
    geom_text(aes(label = paste(n_healed, n_not_healed, sep = "/")),
      hjust = 0, nudge_x = .0025, nudge_y = 7, size = 7.2 / .pt, family = "Arial", colour = "black") +
    scale_size_continuous(range = c(1.6, 2.5), guide = "none") +
    scale_x_continuous(breaks = c(.02, .04, .06, .08, .1), expand = expansion(mult = c(.04, .1))) +
    scale_y_continuous(limits = c(0, 116), breaks = c(0, 25, 50, 75, 100)) +
    labs(x = "Candidate all-cell loading difference", y = "Hypothetical independent total") +
    annotate("text", x = .068, y = 106, label = "Target power 80%; labels = arm sizes",
      size = 7 / .pt, family = "Arial", colour = "black")
  patients <- read_panel_csv("outputs/scientific_revision_20260922/measurement/patient_unit_remap/patient_scores.csv")
  patients$arm <- factor(patients$healing_status, levels = c("healed", "not_healed"), labels = c("Healed", "Not healed"))
  pd <- ggplot(patients, aes(arm, score_cell_weighted, colour = arm, shape = factor(n_samples))) +
    geom_point(position = position_jitter(width = .12, height = 0, seed = 17), size = 2) +
    stat_summary(aes(group = arm), fun = mean, geom = "crossbar", width = .45,
      colour = "black", linewidth = .3) +
    scale_colour_manual(values = c("Healed" = GREEN, "Not healed" = ORANGE), guide = "none") +
    scale_shape_manual(values = c(16, 17), labels = c("One specimen", "Two specimens")) +
    labs(x = "Mapped patient outcomes (7 / 4)", y = "All-cell loading per patient") +
    guides(shape = guide_legend(nrow = 1))
  draw_figure("figure4_outcome_counterexample", list(panel(pa, "Mixture by arm", a),
    panel(pb, "Unit correction and influence", b), panel(pc, "Saved design scenarios", c),
    panel(pd, "Patient all-cell observations", patients)), height = 124, nrow = 2)
}
figure6_sensitivity_negative_controls <- function() {
  s <- historical("method_constant_sensitivity"); n <- historical("pipeline_negative_control")
  raw_control <- read_report("outputs/scientific_revision_20260922/measurement_controls/raw_shuffle_report.json")
  dfu_control <- read_report("outputs/scientific_revision_20260922/measurement_controls/dfu_label_report.json")
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
  dfu_high <- dfu_control$specimen_test$readouts[[2]]
  d <- read_panel_csv("outputs/scientific_revision_20260922/measurement_controls/dfu_specimen_exact_null.csv")
  d <- d[d$readout == "high_state_fraction", ]
  d$observed_difference <- dfu_high$difference
  d$null_low <- dfu_high$null_percentile_025; d$null_high <- dfu_high$null_percentile_975
  pd <- ggplot(d, aes(difference)) +
    annotate("rect", xmin = dfu_high$null_percentile_025, xmax = dfu_high$null_percentile_975,
      ymin = 0, ymax = Inf, fill = "#EDF2F6", alpha = .7) +
    geom_histogram(bins = 26, fill = BLUE, colour = "white", linewidth = .1) +
    geom_vline(xintercept = c(-abs(dfu_high$difference), abs(dfu_high$difference)),
      colour = RED, linewidth = .5, linetype = c("dotted", "solid")) +
    labs(x = "DFU-only label difference", y = "Allocations (n = 2,002)")
  path <- "outputs/scientific_revision_20260922/measurement_controls/raw_shuffle_cell_stats.csv.gz"
  sources[[path]] <<- digest::digest(file = file.path(ROOT, path), algo = "sha256")
  active_sources <<- unique(c(active_sources, path))
  e <- read.csv(gzfile(file.path(ROOT, path)), stringsAsFactors = FALSE)
  e$label <- factor(e$branch, levels = c("real_counts", "column_shuffled_counts"), labels = c("Intact", "Shuffled"))
  pe <- ggplot(e, aes(theta_perplexity, colour = label)) + stat_ecdf(linewidth = .6) +
    scale_colour_manual(values = c("Intact" = TEAL, "Shuffled" = GRAY)) +
    scale_x_continuous(limits = c(1, 15), breaks = c(1, 5, 10, 15)) +
    labs(x = "Cell perplexity (6,000 per branch)", y = "Empirical cumulative fraction") +
    guides(colour = guide_legend(nrow = 1, override.aes = list(linewidth = 1.1))) +
    theme(legend.key.width = unit(5, "mm"))
  f <- data.frame(label = c("Observed p", "Null rate"), value = c(n$N3_permuted_positive_control$p_real, n$N3_permuted_positive_control$fraction_significant_under_permutation))
  pf <- columns(f, "Probability", c(RED, BLUE), c(0, .065)) +
    geom_hline(yintercept = .05, linetype = "dashed", colour = GRAY, linewidth = .3) +
    scale_x_discrete(labels = c("Observed\np", "Null\nrate"))
  draw_figure("figure6_sensitivity_negative_controls", list(panel(pa, "Semantic association", a),
    panel(pb, "Healing contrast", a), panel(pc, "Topic count", k), panel(pd, "Label permutation", d),
    panel(pe, "Structureless input", e), panel(pf, "Permutation reference", f)), nrow = 2, height = 119)
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
              family = "Arial", fontface = "plain", colour = "black") +
    scale_fill_gradient2(low = "#AFC5D6", mid = "#F8FAFB", high = "#E1A1A1", limits = c(-1, 1), name = "ρ") +
    scale_x_continuous(breaks = 1:4, labels = method_names, expand = c(0, 0)) +
    scale_y_reverse(breaks = 1:4, labels = method_names, expand = c(0, 0)) +
    coord_fixed() + labs(x = NULL, y = NULL) +
    theme(axis.line = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank(),
          legend.position = "right", legend.key.height = unit(15, "mm"), legend.key.width = unit(2.4, "mm"),
          legend.title = element_text(size = 8, colour = "black", face = "plain")) + guides(fill = guide_colorbar(display = "rectangles", nbin = 60,
              barheight = unit(28, "mm"), barwidth = unit(2.4, "mm")))
  draw_figure("figure7_representation_inference", list(panel(pa, "Historical pooled-fold AUC", a),
    panel(pc, "Score correlation", c)), height = 71, widths = c(81, 88))
}
figure8_patient_unit_anatomy <- function() {
  r <- read_report("outputs/scientific_revision_20260923/patient_pair_discrimination/report.json")
  t <- repaired("paired_anatomical_control")
  stopifnot(identical(chr(r$summary, "method"), methods))
  a <- data.frame(label = method_names, value = num(r$summary, "auc"),
                  pair_credit = num(r$summary, "credit_sum"), pairs = num(r$summary, "n_pairs"))
  a$label <- factor(a$label, levels = rev(method_names))
  a$wins <- num(r$summary, "wins"); a$ties <- num(r$summary, "ties"); a$losses <- num(r$summary, "losses")
  credits <- do.call(rbind, lapply(c("wins", "ties", "losses"), function(key)
    data.frame(label = a$label, outcome = key, fraction = a[[key]] / a$pairs)))
  pa <- ggplot(a, aes(value, label)) +
    geom_col(data = credits, aes(x = fraction, y = label, fill = outcome), inherit.aes = FALSE,
      orientation = "y", width = .55, position = position_stack(reverse = TRUE)) +
    scale_fill_manual(values = c(wins = "#9FC8CF", ties = "#EEEEEE", losses = "#ECCFCF"),
      breaks = c("wins", "ties", "losses"), labels = c("Win", "Tie", "Loss")) +
    zero_v(.5) + geom_point(size = 2, shape = 18, colour = "black") +
    geom_text(aes(label = paste(wins, ties, losses, sep = "/")), x = .97, hjust = 1,
      colour = "black", family = "Arial", size = 7.2 / .pt) +
    scale_x_continuous(limits = c(0, 1), expand = expansion(mult = .025)) +
    labs(x = "Pair credit; counts = win/tie/loss", y = NULL) + xgrid() + guides(fill = guide_legend(nrow = 1))
  raw <- read_panel_csv("outputs/analysis/paired_anatomical_control/paired_patient_scores.csv")
  b <- rbind(data.frame(patient = raw$patient_id, readout = "All cells",
      value = raw$topic0_all_mean_foot - raw$topic0_all_mean_forearm),
    data.frame(patient = raw$patient_id, readout = "Fibroblasts",
      value = raw$topic0_fibro_mean_foot - raw$topic0_fibro_mean_forearm))
  estimate <- do.call(rbind, lapply(c("topic0_all_mean", "topic0_fibro_mean"), function(key) {
    row <- t$topic0_contrasts[[key]]
    data.frame(readout = if (key == "topic0_all_mean") "All cells" else "Fibroblasts",
      value = row$observed_difference_foot_minus_forearm, low = row$bootstrap_95_ci[[1]], high = row$bootstrap_95_ci[[2]])
  }))
  pb <- ggplot(b, aes(readout, value)) + zero_h() +
    geom_point(position = position_jitter(width = .13, height = 0, seed = 17), size = 1.5, colour = GRAY) +
    geom_errorbar(data = estimate, aes(ymin = low, ymax = high), width = .18, colour = GREEN, linewidth = .6) +
    geom_point(data = estimate, size = 2.1, shape = 18, colour = GREEN) +
    labs(x = "Ten paired patients per readout", y = "Foot minus forearm loading")
  b$low <- NA_real_; b$high <- NA_real_; b$kind <- "patient_difference"
  b <- rbind(b, transform(estimate, patient = NA_character_, kind = "mean_and_interval"))
  pairs <- read_panel_csv("outputs/scientific_revision_20260923/patient_pair_discrimination/pair_scores.csv")
  pairs$method <- factor(pairs$method, levels = rev(methods), labels = rev(method_names))
  pairs$credit <- factor(pairs$credit, levels = c(0, .5, 1), labels = c("Loss", "Tie", "Win"))
  pc <- ggplot(pairs, aes(fold + 1, method, fill = credit)) +
    geom_tile(colour = "white", linewidth = .15) +
    scale_fill_manual(values = c("Loss" = "#ECCFCF", "Tie" = "#EEEEEE", "Win" = "#9FC8CF"),
      breaks = c("Win", "Tie", "Loss"), drop = FALSE) +
    scale_x_continuous(breaks = c(1, 7, 14, 21, 28), expand = c(0, 0)) +
    labs(x = "Dependent held pairs (28; 11 patients)", y = NULL) +
    theme(panel.grid = element_blank(), axis.line = element_blank(), axis.ticks.y = element_blank())
  d <- rbind(data.frame(patient = raw$patient_id, site = "Forearm", readout = "All cells", value = raw$topic0_all_mean_forearm),
    data.frame(patient = raw$patient_id, site = "Foot", readout = "All cells", value = raw$topic0_all_mean_foot),
    data.frame(patient = raw$patient_id, site = "Forearm", readout = "Fibroblasts", value = raw$topic0_fibro_mean_forearm),
    data.frame(patient = raw$patient_id, site = "Foot", readout = "Fibroblasts", value = raw$topic0_fibro_mean_foot))
  d$site <- factor(d$site, levels = c("Forearm", "Foot"))
  pd <- ggplot(d, aes(site, value, group = patient)) +
    geom_line(colour = GRAY, alpha = .7, linewidth = .45) +
    geom_point(aes(shape = site), colour = GREEN, size = 1.8) +
    facet_wrap(~readout, nrow = 1, scales = "free_y") +
    labs(x = "Within-patient tissue pairs (n = 10)", y = "Topic loading") +
    theme(legend.position = "none", strip.background = element_blank(),
      strip.text = element_text(size = 7.5, colour = "black", face = "plain"))
  draw_figure("figure8_patient_unit_anatomy", list(panel(pa, "Same-model held pairs", a),
    panel(pb, "Paired anatomy", b), panel(pc, "Pair-level outcomes", pairs),
    panel(pd, "Individual anatomy pairs", d)), height = 132, nrow = 2)
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
  cohort_labels <- c("GSE223964" = "223964 (n=8)", "GSE245703" = "245703 (n=12)", "GSE268834" = "268834 (n=8)")
  pa <- pa + scale_y_discrete(labels = cohort_labels)
  pb <- pb + scale_y_discrete(labels = cohort_labels)
  samples <- do.call(rbind, lapply(cohort, function(key) {
    rows <- if (key == "GSE268834") g$samples else i$cohorts[[key]]$samples
    data.frame(cohort = key, sample = chr(rows, "gsm"), value = num(rows, "topic0_fibro_mean"),
      immune_fraction = num(rows, "immune_fraction_exclusive"),
      median_genes = num(rows, "median_genes_per_cell"))
  }))
  samples$cohort <- factor(samples$cohort, levels = cohort)
  pd <- ggplot(samples, aes(immune_fraction, value, colour = cohort, shape = cohort)) +
    geom_point(size = 1.5) + scale_colour_manual(values = c(BLUE, ORANGE, TEAL)) +
    scale_shape_manual(values = c(16, 17, 15)) +
    labs(x = "Exclusive immune fraction", y = "Specimen fibroblast mean") +
    guides(colour = guide_legend(nrow = 2), shape = guide_legend(nrow = 2))
  pe <- ggplot(samples, aes(median_genes, value, colour = cohort, shape = cohort)) +
    geom_point(size = 1.5) + scale_colour_manual(values = c(BLUE, ORANGE, TEAL)) +
    scale_shape_manual(values = c(16, 17, 15)) + labs(x = "Median detected genes per cell", y = "Specimen fibroblast mean") +
    theme(legend.position = "none")
  groups <- do.call(rbind, lapply(c("GSE268834", "GSE248247"), function(key) {
    rows <- if (key == "GSE268834") g$samples else h$samples
    data.frame(cohort = key, sample = chr(rows, "gsm"), group = chr(rows, "group"), value = num(rows, "topic0_fibro_mean"))
  }))
  groups$group <- factor(groups$group, levels = c("non_diabetic", "diabetic"), labels = c("Non-D", "D"))
  pf <- ggplot(groups, aes(group, value, colour = group)) +
    geom_point(position = position_jitter(width = .07, height = 0, seed = 17), size = 1.7) +
    facet_wrap(~cohort, nrow = 1) + scale_colour_manual(values = c(PURPLE, GREEN)) +
    labs(x = "Source-labelled specimen groups", y = "Fibroblast mean loading") +
    theme(legend.position = "none", strip.background = element_blank(),
      strip.text = element_text(size = 7.2, colour = "black", face = "plain"))
  draw_figure("figure9_external_construct_audit", list(panel(pa, "Immune context", a),
    panel(pb, "Detected genes", b), panel(pc, "Group contrast", c), panel(pd, "Immune-axis observations", samples),
    panel(pe, "Depth-axis observations", samples), panel(pf, "External group values", groups)),
    height = 139, nrow = 2, widths = c(59, 59, 46))
}


source(file.path(ROOT, "manuscripts/r_tikz/biological_figures.R"))
names_all <- c("figure1_workflow", "figure2_mixture_semantics", "figure3_artifact_controls", "figure4_outcome_counterexample", "figure5_patient_readouts", "figure6_sensitivity_negative_controls", "figure7_representation_inference", "figure8_patient_unit_anatomy", "figure9_external_construct_audit")
selected <- if (length(requested)) requested else names_all
if (!all(selected %in% names_all)) stop("Unknown figure name")
for (name in selected) get(name, mode = "function")()
manifest <- list(renderer = "R + ggplot2/grid + tikzDevice + XeLaTeX", font = "Arial",
  text_colour = "#000000", text_weight = "regular", panel_label_weight = "bold",
  panel_title_alignment = "panel-centred",
  base_font_pt = 8, tick_font_pt = 7.5, label_font_pt = 11, journal_width_mm = 180,
  figures = figures, sources = sources, R = R.version.string,
  packages = lapply(c("ggplot2", "tikzDevice", "jsonlite", "digest"), function(p) list(package = p, version = as.character(packageVersion(p)))),
  design_sources = setNames(lapply(c("construct_design.tikz"),
    function(f) digest::digest(file = file.path(ROOT, "manuscripts/r_tikz", f), algo = "sha256")),
    paste0("manuscripts/r_tikz/", c("construct_design.tikz"))),
  script_sha256 = digest::digest(file = normalizePath(script), algo = "sha256"))
auxiliary <- c("manuscripts/r_tikz/biological_figures.R", "manuscripts/r_tikz/quality_figures.R", "manuscripts/r_tikz/followup_figures.R", "manuscripts/r_tikz/panel_a.tex")
manifest$auxiliary_sources <- setNames(lapply(auxiliary,function(f) digest::digest(file=file.path(ROOT,f),algo="sha256")),auxiliary)
write_json(manifest, file.path(BUILD, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(capture.output(sessionInfo()), file.path(BUILD, "sessionInfo.txt"))

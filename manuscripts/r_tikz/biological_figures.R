# Publication panels from saved biological-expansion outputs. No inference here.
bio_csv <- function(study, name, extension="biological_expansion") {
  path <- paste0("outputs/", extension, "/", study, "/", name, ".csv")
  full <- file.path(ROOT, path)
  sources[[path]] <<- digest::digest(file = full, algo = "sha256")
  active_sources <<- unique(c(active_sources, path))
  read.csv(full, stringsAsFactors = FALSE)
}
LINEAGE <- c(fibroblast="Fibro.", keratinocyte="Keratin.", myeloid="Myeloid",
  t_cell="T cell", b_plasma="B/plasma", endothelial="Blood EC", lymphatic_ec="Lymph EC",
  pericyte_smc="Mural", mast="Mast", melanocyte="Melano.", sweat_gland="Gland")
LINEAGE_COL <- setNames(c(TEAL, ORANGE, PURPLE, BLUE, "#CE79A0", RED,
                         "#B0A043", "#5E6B81", "#7DA14E", "#6F4E37", "#ADAAA5"), names(LINEAGE))
TIME_LABEL <- c(Skin="Day 0",Wound1="Day 1",Wound7="Day 7",Wound30="Day 30")
TIME_COL <- c(Skin=GRAY,Wound1=BLUE,Wound7=ORANGE,Wound30=TEAL)
small_map <- function(p) p + theme(panel.grid=element_blank(), axis.text=element_blank(),
  axis.ticks=element_blank(), axis.line=element_blank(), legend.key.size=unit(2.5,"mm"))
foot_groups <- function(composition) {
  spec <- unique(composition[, c("gsm", "arm")])
  spec$arm <- factor(c("DFU-healer"="Healer","DFU-nonhealer"="Non-healer","Non-diabetic"="Healthy")[spec$arm], levels=names(ARMS))
  spec <- spec[order(spec$arm, spec$gsm), ]
  spec$slot <- ave(seq_len(nrow(spec)), spec$arm, FUN=seq_along)
  spec$column <- (spec$slot - 1) %% 3
  spec$row <- (spec$slot - 1) %/% 3
  spec
}
acute_schedule <- function() {
  expand.grid(donor=factor(c("Donor C","Donor B","Donor A"), levels=c("Donor C","Donor B","Donor A")),
              day=c(0, 1, 7, 30), stringsAsFactors=FALSE)
}

# Display summaries only; no refitting, outcome testing or new resampling.
coordinate_summaries <- function(data, group) {
  do.call(rbind, lapply(split(data, data[[group]]), function(part) {
    data.frame(group = part[[group]][1], x = median(part$x), y = median(part$y),
      xlow = quantile(part$x, .25), xhigh = quantile(part$x, .75),
      ylow = quantile(part$y, .25), yhigh = quantile(part$y, .75), n_display = nrow(part),
      stringsAsFactors = FALSE, row.names = NULL)
  }))
}
coordinate_view <- function(plot, data, xlab, ylab, xmax = 4.3, inset = c(.71, .64, 1, 1)) {
  full <- plot + labs(x = NULL, y = NULL) +
    scale_x_continuous(breaks = c(0, 10)) + scale_y_continuous(breaks = c(0, 10)) +
    theme(legend.position = "none", panel.grid = element_blank(), plot.margin = margin(1,1,1,1,"mm"),
      axis.text = element_text(size = 7, colour = "black"), axis.ticks.length = unit(.6, "mm"),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.border = element_rect(fill = NA, colour = GRAY, linewidth = .25))
  focused <- plot + coord_cartesian(xlim = c(-4.7, xmax), ylim = c(-3.2, 4.8)) +
    labs(x = xlab, y = ylab)
  focused + inset_element(full, left = inset[1], bottom = inset[2], right = inset[3], top = inset[4], align_to = "panel")
}

biological_measurement_figure <- function() {
  d <- read_report("outputs/biological_expansion/measurement/report.json")
  f <- bio_csv("measurement","cell_composition")
  a <- foot_groups(f)
  pa <- structure(list(asset="panel_a"),class="vector_panel")
  b <- bio_csv("measurement","atlas_display")
  pb <- small_map(ggplot(b,aes(x,y,colour=celltype))+geom_point(size=.24,alpha=.8)+
    scale_colour_manual(values=LINEAGE_COL,labels=LINEAGE)+labs(x="UMAP 1",y="UMAP 2")+
    guides(colour=guide_legend(ncol=2,byrow=TRUE,override.aes=list(size=1.5,alpha=1))))
  pc <- small_map(ggplot(b,aes(x,y,colour=topic_0))+geom_point(size=.28)+
    scale_colour_gradient(low="#DDE1E5",high=TEAL,name="Loading",limits=c(0,1))+
    labs(x="UMAP 1",y="UMAP 2")+guides(colour=guide_colourbar(display="rectangles",title.position="top",barwidth=unit(25,"mm"),barheight=unit(2,"mm"))))
  c <- bio_csv("measurement","marker_expression")
  c$celltype <- factor(c$celltype,levels=rev(names(LINEAGE)),labels=rev(unname(LINEAGE)))
  c$gene <- factor(c$gene,levels=unique(c$gene))
  c$scaled <- ave(c$mean_log1p_cp10k,c$gene,FUN=function(x) (x-min(x))/max(diff(range(x)),1e-10))
  pd <- ggplot(c,aes(celltype,gene,size=detected_fraction,colour=scaled))+geom_point()+
    scale_size_continuous(range=c(.15,2.1),breaks=c(.25,.75),labels=c("25%","75%"))+
    scale_colour_gradient(low="#D3DAE0",high=TEAL,breaks=c(0,1))+
    labs(x=NULL,y=NULL)+guides(size=guide_legend(title="Detected",order=1),colour=guide_colourbar(display="rectangles",title="Scaled mean",order=2,barwidth=unit(12,"mm"),barheight=unit(2,"mm")))+
    theme(axis.text.x=element_text(angle=90,hjust=1,vjust=.5),panel.grid.major.y=element_blank(),
      legend.box="vertical",legend.spacing.y=unit(.6,"mm"),legend.title=element_text(size=7,face="plain",colour="black"))
  e <- bio_csv("measurement","loading_distribution")
  e$arm <- factor(c("DFU-healer"="Healer","DFU-nonhealer"="Non-healer","Non-diabetic"="Healthy")[e$arm], levels=names(ARMS))
  pe <- ggplot(e,aes(loading,density,colour=arm))+geom_step(linewidth=.5)+zero_v(d$threshold)+
    scale_colour_manual(values=ARMS)+labs(x="Loading within fibroblasts",y="Density")+
    scale_x_continuous(limits=c(0,1),breaks=c(0,.25,.5,.75,1))+theme(legend.position="bottom",legend.key.width=unit(5,"mm"))+
    guides(colour=guide_legend(nrow=1,override.aes=list(linewidth=1.1)))
  tail <- pe + coord_cartesian(ylim = c(0, 2.5)) + labs(x = NULL, y = NULL) +
    theme(legend.position = "none", panel.grid = element_blank(),
      panel.border = element_rect(fill = NA, colour = GRAY, linewidth = .25))
  pe <- pe + inset_element(tail, left = .40, bottom = .33, right = .97, top = .96, align_to = "panel")
  f$arm <- factor(c("DFU-healer"="Healer","DFU-nonhealer"="Non-healer","Non-diabetic"="Healthy")[f$arm],levels=names(ARMS))
  pf <- ggplot(f,aes(gsm,fraction,fill=celltype))+geom_col(width=.9)+
    scale_fill_manual(values=LINEAGE_COL)+facet_grid(~arm,scales="free_x",space="free_x")+
    scale_y_continuous(breaks=c(0,.5,1))+labs(x="Individual specimens",y="Recovered-cell fraction")+
    theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),legend.position="none",strip.background=element_blank(),strip.text=element_text(size=7.5,face="plain",colour="black"))
  draw_figure("figure1_workflow",list(panel(pa,"Foot-skin sampling and the measurement question",a),
    panel(pb,"Recovered cells",b),panel(pc,"Cell loading",b),
    panel(pd,"Marker expression",c),panel(pe,"Fibroblast loading distributions",e),
    panel(pf,"Specimen composition",f)),height=212,nrow=3,layout=physical_object_layout())
}

figure5_patient_readouts <- function() {
  d <- bio_csv("measurement","patient_readouts")
  d <- d[d$arm!="Non-diabetic",]
  d$arm <- factor(c("DFU-healer"="Healed","DFU-nonhealer"="Not healed")[d$arm],levels=c("Healed","Not healed"))
  rs <- c("fibroblast_mean","high_state_fraction","fibroblast_fraction")
  titles <- c("Fibroblast mean","High-state fraction","Fibroblast composition")
  panels <- lapply(seq_along(rs),function(i) {
    x <- data.frame(patient=d$patient,arm=d$arm,value=d[[rs[i]]])
    p <- ggplot(x,aes(arm,value,colour=arm))+geom_point(position=position_jitter(width=.12,seed=17),size=2)+
      stat_summary(fun=mean,geom="crossbar",width=.45,colour="black",linewidth=.3)+
      scale_colour_manual(values=c("Healed"=GREEN,"Not healed"=ORANGE))+
      labs(x="Patient outcome",y=c("Mean loading","Fraction above fixed cut","Fraction of recovered cells")[i])+
      theme(legend.position="none")
    panel(p,titles[i],x)
  })
  extra <- read_report("outputs/computational_extension/measurement/report.json")
  cuts <- bio_csv("measurement","threshold_sensitivity","computational_extension")
  p <- ggplot(cuts,aes(threshold,difference))+zero_h()+geom_line(colour=TEAL,linewidth=.5)+
    geom_point(aes(shape=is_original),size=1.7)+scale_shape_manual(values=c(16,17),labels=c("Grid","Original cut"))+
    labs(x="High-state threshold",y="Healed minus not healed")+guides(shape=guide_legend(nrow=1))
  panels[[4]] <- panel(p,"Threshold sensitivity",cuts)
  sample <- bio_csv("measurement","sampling_summary","computational_extension")
  for (i in 1:2) {
    x <- sample[sample$sensitivity==c("Equal 100 cells per specimen","One specimen per patient")[i],]
    x$readout <- factor(x$readout,levels=rev(rs),labels=rev(c("Programme mean","High-state fraction","Fibroblast fraction")))
    p <- ggplot(x,aes(difference_mean,readout))+zero_v()+
      geom_errorbar(aes(xmin=difference_min,xmax=difference_max),orientation="y",width=.18,colour=NAVY,linewidth=.5)+
      geom_point(size=1.8,colour=NAVY)+scale_x_continuous(breaks=c(0,.1,.2))+
      labs(x="Healed minus not healed",y=NULL)+xgrid()
    draws <- bio_csv("measurement", c("equal_cell_draws", "one_specimen_choices")[i], "computational_extension")
    draws$readout <- factor(draws$readout, levels = rev(rs), labels = rev(c("Programme mean", "High-state fraction", "Fibroblast fraction")))
    p <- p + geom_point(data = draws, aes(x = difference, y = readout), inherit.aes = FALSE,
      position = position_jitter(width = 0, height = .13, seed = 17), size = if (i == 1) .6 else 1.2,
      alpha = if (i == 1) .25 else .65, colour = ORANGE)
    x <- merge(x, draws, by = "readout")
    panels[[i+4]] <- panel(p,c("Equal cell counts","One specimen per patient")[i],x)
  }
  evidence <- read_report("outputs/biological_expansion/measurement/report.json")
  effects <- do.call(rbind,lapply(Filter(function(x) x$variant=="cell-weighted",evidence$contrasts),as.data.frame))
  effects$readout <- factor(effects$readout,levels=rev(rs),labels=rev(titles))
  pg <- ggplot(effects,aes(difference,readout))+zero_v()+
    geom_errorbar(aes(xmin=ci_low,xmax=ci_high),orientation="y",width=.17,colour=NAVY,linewidth=.6)+
    geom_point(colour=NAVY,size=2)+labs(x="Patient contrast with 95% interval",y=NULL)+xgrid()
  panels[[7]] <- panel(pg,"Effect uncertainty",effects)
  omitted <- do.call(rbind,lapply(evidence$omissions,as.data.frame))
  omitted$readout <- factor(omitted$readout,levels=rev(rs),labels=rev(titles))
  ph <- ggplot(omitted,aes(difference,readout))+zero_v()+
    geom_point(position=position_jitter(width=0,height=.08,seed=17),size=1.6,colour=ORANGE)+
      labs(x="Leave-one-patient-out contrast",y=NULL)+xgrid()
  panels[[8]] <- panel(ph,"Patient influence",omitted)
  draw_figure("figure5_patient_readouts",panels,height=162,nrow=4)
}


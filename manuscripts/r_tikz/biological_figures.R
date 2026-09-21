# Publication panels from saved biological-expansion outputs. No inference here.
bio_csv <- function(study, name) {
  path <- paste0("outputs/biological_expansion/", study, "/", name, ".csv")
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
bio_text <- function(...) annotate("text", ..., family="Arial",fontface="bold",colour="black",size=2.65)

skin_material <- function(description) {
  # Schematic anatomy, deliberately separated from measured data panels.
  dermis <- data.frame(x=c(0,10,10,0),y=c(1.8,1.8,6.9,6.9))
  epidermis <- data.frame(x=c(0,3.8,4.3,3.4,0,6.6,5.7,6.2,10,10),
                         y=c(6.9,6.9,6.1,7.9,7.9,7.9,6.1,6.9,6.9,7.9),piece=rep(1:2,each=5))
  fibro <- do.call(rbind,lapply(seq_len(7),function(i) {
    x=c(1.3,3.1,5.2,7.6,8.7,2.1,6.8)[i];y=c(5.9,4.7,5.0,5.8,3.2,2.7,2.6)[i]
    data.frame(x=x+c(-.55,0,.55,0),y=y+c(-.10,.16,.10,-.16),id=i)
  }))
  immune <- data.frame(x=c(4.6,5.4,4.9,6.0),y=c(6.0,6.3,5.5,5.7))
  ggplot()+geom_polygon(data=dermis,aes(x,y),fill="#F6E9DC",colour="#BDB2A7",linewidth=.3)+
    geom_polygon(data=epidermis,aes(x,y,group=piece),fill="#E7C49B",colour="#9B8068",linewidth=.3)+
    geom_polygon(data=fibro,aes(x,y,group=id),fill=TEAL)+
    geom_segment(aes(x=.1,xend=9.9,y=3.9,yend=3.9),colour="#DBAAA1",linewidth=2.7)+
    geom_segment(aes(x=.1,xend=9.9,y=3.9,yend=3.9),colour="white",linewidth=.8)+
    geom_point(data=immune,aes(x,y),colour=PURPLE,size=1.7)+
    bio_text(x=1.65,y=8.6,label="Epidermis")+
    bio_text(x=6.9,y=8.65,label="Wound bed (schematic)")+
    bio_text(x=8.6,y=4.65,label="Dermis")+
    bio_text(x=1.8,y=5.2,label="Fibroblasts")+
    bio_text(x=6.4,y=7.2,label="Immune cells")+
    bio_text(x=7.8,y=3.35,label="Vessel")+
    bio_text(x=5,y=.8,label=description)+
    coord_cartesian(xlim=c(-.2,10.2),ylim=c(-.1,9.25),clip="off")+
    theme_void(base_family="Arial")+theme(plot.margin=margin(0,1,0,1,"mm"))
}

biological_measurement_figure <- function() {
  d <- read_report("outputs/biological_expansion/measurement/report.json")
  a <- data.frame(material="Selected foot-skin specimens",cells=d$cells,specimens=d$specimens,patients=d$patients)
  pa <- skin_material("25 specimens / 20 mapped patients\nDissociation and single-cell RNA sequencing")
  b <- bio_csv("measurement","atlas_display")
  pb <- small_map(ggplot(b,aes(x,y,colour=celltype))+geom_point(size=.24,alpha=.8)+
    scale_colour_manual(values=LINEAGE_COL,labels=LINEAGE)+labs(x="UMAP 1",y="UMAP 2")+
    guides(colour=guide_legend(ncol=3,byrow=TRUE,override.aes=list(size=1.5,alpha=1))))
  pc <- small_map(ggplot(b,aes(x,y,colour=topic_0))+geom_point(size=.28)+
    scale_colour_gradient(low="#DDE1E5",high=TEAL,name="Loading",limits=c(0,1))+
    labs(x="UMAP 1",y="UMAP 2")+guides(colour=guide_colourbar(display="rectangles",title.position="top",barwidth=unit(25,"mm"),barheight=unit(2,"mm"))))
  c <- bio_csv("measurement","marker_expression")
  c$celltype <- factor(c$celltype,levels=rev(names(LINEAGE)),labels=rev(unname(LINEAGE)))
  c$gene <- factor(c$gene,levels=unique(c$gene))
  c$scaled <- ave(c$mean_log1p_cp10k,c$gene,FUN=function(x) (x-min(x))/max(diff(range(x)),1e-10))
  pd <- ggplot(c,aes(gene,celltype,size=detected_fraction,colour=scaled))+geom_point()+
    scale_size_continuous(range=c(.15,2.1),breaks=c(.25,.75),labels=c("25%","75%"))+
    scale_colour_gradient(low="#D3DAE0",high=TEAL,breaks=c(0,1))+
    labs(x=NULL,y=NULL)+guides(size=guide_legend(title="Detected",order=1),colour=guide_colourbar(display="rectangles",title="Scaled mean",order=2,barwidth=unit(12,"mm"),barheight=unit(2,"mm")))+
    theme(axis.text.x=element_text(angle=90,hjust=1,vjust=.5),panel.grid.major.y=element_blank(),legend.title=element_text(size=7,face="bold",colour="black"))
  e <- bio_csv("measurement","loading_distribution")
  e$arm <- c("DFU-healer"="Healer","DFU-nonhealer"="Non-healer","Non-diabetic"="Healthy")[e$arm]
  pe <- ggplot(e,aes(loading,density,colour=arm))+geom_step(linewidth=.5)+zero_v(d$threshold)+
    scale_colour_manual(values=ARMS)+labs(x="Loading within fibroblasts",y="Density")+
    scale_x_continuous(limits=c(0,1),breaks=c(0,.25,.5,.75,1))+theme(legend.position="bottom")
  f <- bio_csv("measurement","cell_composition")
  f$arm <- factor(c("DFU-healer"="Healer","DFU-nonhealer"="Non-healer","Non-diabetic"="Healthy")[f$arm],levels=names(ARMS))
  pf <- ggplot(f,aes(gsm,fraction,fill=celltype))+geom_col(width=.9)+
    scale_fill_manual(values=LINEAGE_COL)+facet_grid(~arm,scales="free_x",space="free_x")+
    scale_y_continuous(breaks=c(0,.5,1))+labs(x="Individual specimens",y="Recovered-cell fraction")+
    theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),legend.position="none",strip.background=element_blank(),strip.text=element_text(size=7.5,face="bold",colour="black"))
  draw_figure("figure1_workflow",list(panel(pa,"Tissue and sampling",a),
    panel(pb,"Observed cell landscape",b),panel(pc,"Programme loading in cells",b),
    panel(pd,"Measured marker expression",c),panel(pe,"Fibroblast loading distributions",e),
    panel(pf,"Specimen composition",f)),height=204,nrow=3)
}

figure5_patient_readouts <- function() {
  d <- bio_csv("measurement","patient_readouts")
  d <- d[d$arm!="Non-diabetic",]
  d$arm <- factor(c("DFU-healer"="Healed","DFU-nonhealer"="Not healed")[d$arm],levels=c("Healed","Not healed"))
  rs <- c("fibroblast_mean","high_state_fraction","fibroblast_fraction")
  titles <- c("Fibroblast programme mean","High-state fraction","Fibroblast composition")
  panels <- lapply(seq_along(rs),function(i) {
    x <- data.frame(patient=d$patient,arm=d$arm,value=d[[rs[i]]])
    p <- ggplot(x,aes(arm,value,colour=arm))+geom_point(position=position_jitter(width=.12,seed=17),size=2)+
      stat_summary(fun=mean,geom="crossbar",width=.45,colour="black",linewidth=.3)+
      scale_colour_manual(values=c("Healed"=GREEN,"Not healed"=ORANGE))+
      labs(x="Patient outcome",y=c("Mean loading","Fraction above fixed cut","Fraction of recovered cells")[i])+
      theme(legend.position="none")
    panel(p,titles[i],x)
  })
  draw_figure("figure5_patient_readouts",panels,height=75)
}


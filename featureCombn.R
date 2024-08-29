args <- commandArgs(TRUE)

if(length(args) !=2){
	print("R script topn base.auc.dir")
	quit()
}

# Initialize variables
top_n <- NULL
base_auc_dir <- NULL

# Parse named arguments
for (i in seq(1, length(args), by=2)) {
  if (args[i] == "--topn") {
    top_n <- args[i + 1]
  } else if (args[i] == "--base.auc.dir") {
    base_auc_dir <- args[i + 1]
  } 
}

if (is.null(top_n) || is.null(base_auc_dir)) {
  print("Usage: Rscript featureCombn.R --topn <topn> --base.auc.dir <base_auc_dir>")
  quit()
}

n = top_n
wdir = base.auc.dir
setwd(wdir)

dir.create('BasemodelScore')
dir.create('StackScore')
dir.create('CombnModel')

library(plyr)

StackTopModelScore <- function(top.bm,model.name){
  top.score <- data.frame()
  for(i in 1:nrow(top.bm)){
    mdir <- paste0("automl/",top.bm[i,"Feature"],"/",top.bm[i,"ModelID"],".Predict.tsv")
    bs.score <- read.table(mdir,header = T,sep = "\t",stringsAsFactors = F)
    bs.score$Feature <- top.bm[i,"Feature"]
    bs.score$ModelID <- top.bm[i,"ModelID"]
    top.score <- rbind(top.score,bs.score)
  }
  write.table(top.score,file = paste0("BasemodelScore/",model.name,".basemodelscore.Predict.tsv"),
              col.names = T,row.names = F,sep = "\t",quote = F)	
  top.score <- ddply(top.score,"SampleID",transform,MeanScore = mean(Score))
  top.meanscore <- unique(top.score[,c("SampleID","PredType","MeanScore")])
  colnames(top.meanscore)[3] <- "Score"
  write.table(top.meanscore,file = paste0("StackScore/",model.name,".meanscore.Predict.tsv"),
              col.names = T,row.names = F,sep = "\t",quote = F)
}


bm.all.auc <- read.table("basemodel.auc.summary.txt",header = T,sep = "\t",stringsAsFactors = F,check.names = F)

feat.list <- list.files("automl")
featureCombn <- function(topn){
  for(n in 3:length(feat.list)){
    com.list <- combn(feat.list,n)
    for(i in 1:ncol(com.list)){
      fl = com.list[, i]
      top.auc = data.frame()
      for(j in 1:length(fl)){
        feature.auc = bm.all.auc[bm.all.auc$Feature == fl[j], ]
        feature.auc = feature.auc[order(feature.auc$AUC,decreasing = T), ]
        feature.auc.top = rbind(head(feature.auc[feature.auc$Group2=="Train", ],topn))
        top.auc <- rbind(top.auc,feature.auc.top)
      }
      StackTopModelScore(top.auc,paste(c(fl,topn),collapse = "-"))
      write.table(top.auc,file = paste0('CombnModel/',paste(c(fl,topn),collapse = "-"),'.auc.txt'),row.names = F,col.names = T,sep = "\t",quote = F)
    }
  }
}

featureCombn(as.numeric(n))

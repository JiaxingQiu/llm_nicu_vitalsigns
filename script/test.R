rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../")
print(getwd())

library(readxl)
library(ggplot2)
library(reshape2)

view_k_row <- function(ts, k_row=10, i_row=NULL, vs = c("HR","SP", "XC")[1]){
  ts_sub <- ts[1:k_row, 3:ncol(ts)]
  if(!is.null(i_row)){
    ts_sub <- ts[i_row, 3:ncol(ts)]
  }
  ts_sub$row <- seq_len(nrow(ts_sub))
  ts_sub_melted <- melt(ts_sub, id.vars = "row")
  p <- ggplot(ts_sub_melted, aes(x = variable, y = value, group = row, color = as.factor(row))) +
    geom_line() 
  if(vs == "HR") {
    p <- ggplot(ts_sub_melted, aes(x = variable, y = value, group = row)) +
      geom_line(color = "#F8766D") +
      geom_hline(aes(yintercept=80),size=0.5,linetype="dashed", color="darkgray") +
      geom_hline(aes(yintercept=90),size=0.5,linetype="dashed", color="gray")
      
  }
  if(vs == "SP") {
    p <- ggplot(ts_sub_melted, aes(x = variable, y = value, group = row)) +
      geom_line(color = "#619CFF") +
      geom_hline(aes(yintercept=80),size=0.5,linetype="dashed", color="darkgray") +
      geom_hline(aes(yintercept=90),size=0.5,linetype="dashed", color="gray")
    
  }
  if(vs == "XC"){
    xc = round(as.numeric(cor(unlist(ts_sub[1,]),unlist(ts_sub[2,]))),2)
    p <- p + labs(x = "Time (2s)", color = NULL) +
      theme_classic() +
      scale_color_manual(values = c("#F8766D", "#619CFF"))+
      labs(subtitle = paste0("cross-correlation = ",xc))+
      theme(legend.position = "none",
            axis.text.x = element_blank())
  }
  
  return(p)
}


ts_hr <- read_excel("./data/PAS Challenge HR Data.xlsx")
ts_sp <- read_excel("./data/PAS Challenge SPO2 Data.xlsx")
view_k_row(ts_hr)
view_k_row(ts_sp)

# in df ts_hr, find rows where any cell under 80
rows_under_80 <- apply(ts_hr, 1, function(x) any(x < 80))
ts_hr_b80 <- ts_hr[rows_under_80, ]
view_k_row(ts_hr_b80, i_row = 2, vs = 'HR')

rows_under_90 <- apply(ts_sp, 1, function(x) any(x < 90))
ts_sp_b90 <- ts_sp[rows_under_90, ]
view_k_row(ts_sp_b90, i_row = 1, vs = 'SP')



res_df = ts_hr[,c("VitalID", "VitalTime")]
res_df$xc <- NA
for(id in unique(ts_hr$VitalID)){
  for(tt in unique(ts_hr$VitalTime[which(ts_hr$VitalID==id)])){
    hr = unlist(ts_hr[which(ts_hr$VitalID==id & ts_hr$VitalTime==tt),3:ncol(ts_hr)])
    sp = unlist(ts_sp[which(ts_sp$VitalID==id & ts_sp$VitalTime==tt),3:ncol(ts_sp)])
    xc = as.numeric(cor(hr, sp))
    res_df[which(res_df$VitalID==id & res_df$VitalTime==tt),'xc'] <- xc
  }
}
res_df_high_xc <- res_df[which(res_df$xc>0.8),]

ts_xc = rbind(ts_hr[ts_hr$VitalID==1031 & ts_hr$VitalTime==6128566,],
              ts_sp[ts_sp$VitalID==1031 & ts_sp$VitalTime==6128566,])
view_k_row(ts_xc,k_row=2,vs="XC")

ts_xc = rbind(ts_hr[ts_hr$VitalID==3430 & ts_hr$VitalTime==482052,],
              ts_sp[ts_sp$VitalID==3430 & ts_sp$VitalTime==482052,])
view_k_row(ts_xc,k_row=2,vs="XC")




length(unique(ts_hr$VitalID))








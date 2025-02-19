rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../")
print(getwd())

# --- load packages ---
library(readxl)
library(ggplot2)
library(reshape2)

# --- load functions ---
path = paste0("./script/utils")
flst = list.files(path)
sapply(c(paste(path,flst,sep="/")), source, .GlobalEnv)


# --- load 10-minutes records ---
ts_hr <- read_excel("./data/PAS Challenge HR Data.xlsx")
ts_sp <- read_excel("./data/PAS Challenge SPO2 Data.xlsx")
view_k_row(ts_hr) # viz first 10 rows
view_k_row(ts_sp)

# --- load patient demo ---
df_outc <- read_excel("./data/PAS Challenge Outcome Data.xlsx")
df_demo <- read_excel("./data/PAS Challenge Demographic Data.xlsx")


# --- brady 80 (heart rate < 80) ---
th <- 80
rows <- apply(ts_hr[,3:ncol(ts_hr)], 1, function(x) any(x < th))
ts_sub <- ts_hr[rows,3:ncol(ts_hr)]
view_k_row(ts_sub, i_row = 2, vs = 'HR')
# apply function threshold_events(x, th, "<") to each row (x) of dataframe ts_sub
events <- apply(ts_sub, 1, function(x) threshold_event(x, th = th, direction = "<"))
events <- apply(ts_sub, 1, function(x) threshold_event_extra(x, th = th, direction = "<"))
threshold_event_extra(x=as.numeric(unlist(ts_sub[2,])), th = 80, direction = "<", plot=T)
threshold_event_extra(x=as.numeric(unlist(ts_sub[2,])), th = 90, direction = "<", plot=T)


# --- hyperoxia (SPO2 > 98) ---
th <- 98
rows <- apply(ts_sp[,3:ncol(ts_sp)], 1, function(x) any(x > th))
ts_sub <- ts_sp[rows,3:ncol(ts_sp)]
view_k_row(ts_sub, i_row = 2, vs = 'SP')
events <- apply(ts_sub, 1, function(x) threshold_event(x, th = th, direction = ">"))
# events <- apply(ts_sp[,3:ncol(ts_sp)], 1, function(x) threshold_event(x, th = th, direction = ">"))

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


ts_sub <- ts_hr[,3:ncol(ts_hr)]
events80 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 80, plot=F, type = 0))
events90 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 90, plot=F, type = 0))
events100 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 100, plot=F, type = 0))


get_most_severe_event <- function(row_index){
  # for each row of ts_sub
  # find the most severe event in events80, events90 and events100
  # keep the most severe event
  # if no event, return "no event." 
  event80 <- events80[row_index]
  event90 <- events90[row_index]
  event100 <- events100[row_index]

  if(all(c(event80, event90, event100) == "")){
    return(NA)
  }
  if(event80!=""){
    return(event80)
  }
  if(event90!=""){
    return(event90)
  }
  if(event100!=""){
    return(event100)
  }
}

# apply the function to each row_index of ts_sub
most_severe_events <- sapply(seq(1,nrow(ts_sub)), get_most_severe_event)
# count NA in most_severe_events
sum(!is.na(most_severe_events))
# get row index of not NA in most_severe_events
row_index <- which(!is.na(most_severe_events))
# ts_event dataframe
ts_event <- ts_hr[row_index, ]
# get values of most_severe_events for the row_index
ts_event_description <- most_severe_events[row_index]
ts_event$event_description <- ts_event_description

write.csv(ts_event, "./data/HR_events.csv", row.names = F)



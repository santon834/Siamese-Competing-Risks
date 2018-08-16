# imports
library("survival")
library("pec")
rm(list=ls())

c_td = vector(mode = "numeric", length=5)
typ = 3 #1/2/3

for(no in 0:(4)){
  train_feature_values <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/train_feature_values_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  train_c_time <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/train_c_surv_time_x_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  train_uc_time <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/train_uc_surv_time_x_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  train_c_no = dim(train_c_time)[1]
  train_uc_no = dim(train_uc_time)[1]
  train_time = as.matrix(rbind(train_c_time, train_uc_time))
  train_status = rbind(matrix(0, train_c_no, 1), matrix(1, train_uc_no, 1))
  
  val_feature_values <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/val_feature_values_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  val_c_time <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/val_c_surv_time_x_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  val_uc_time <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/val_uc_surv_time_x_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  val_c_no = dim(val_c_time)[1]
  val_uc_no = dim(val_uc_time)[1]
  val_time = as.matrix(rbind(val_c_time, val_uc_time))
  val_status = rbind(matrix(0, val_c_no, 1), matrix(1, val_uc_no, 1))
  
  test_feature_values <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/test_feature_values_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  test_c_time <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/test_c_surv_time_x_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  test_uc_time <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER/test_uc_surv_time_x_', no, '_', typ, '.csv'), header = FALSE, sep=',')
  test_c_no = dim(test_c_time)[1]
  test_uc_no = dim(test_uc_time)[1]
  test_time = as.matrix(rbind(test_c_time, test_uc_time))
  test_status = rbind(matrix(0, test_c_no, 1), matrix(1, test_uc_no, 1))
  
  train_time = rbind(train_time, val_time)
  train_status = rbind(train_status, val_status)
  train_feature_values = rbind(train_feature_values ,val_feature_values)
    
  S <- Surv(time = train_time , event = train_status, type='right')
  cox_model <- coxph(S ~ ., data=train_feature_values)

  preds = 1.-t(survfit(cox_model, test_feature_values)$surv)
  patient_no = nrow(test_feature_values)
  total_pair = 0.0
  cor_pair = 0.0
  for(i in 1:patient_no){
    if(test_status[i,1] == 1){
      T1 = test_time[i]+1
      R1 = preds[i,T1]
      for(j in 1:patient_no){
        T2 = test_time[j]+1
        R2 = preds[j,T1]
        if(T1 < T2){
          total_pair = total_pair+1.
          if(R1 > R2){cor_pair = cor_pair+1.}}}}}
  c_td[no+1] = cor_pair/total_pair
  
}
c_td = unlist(c_td)

print(mean(c_td))
print(mean(c_td)+sd(c_td)*1.96)
print(mean(c_td)-sd(c_td)*1.96)
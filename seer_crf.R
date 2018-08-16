# imports
library("survival")
library("cmprsk")
library("pec")
rm(list=ls())

c_td1 = vector(mode = "numeric", length=5)
c_td2 = vector(mode = "numeric", length=5)
c_td3 = vector(mode = "numeric", length=5)

for(no in 0:(4)){
  train_feature_values <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_feature_values_', no, '.csv'), header = FALSE, sep=',')
  train_time <- as.matrix(read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_surv_time_x_', no, '.csv'), header = FALSE, sep=','))
  train_status <- as.matrix(read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_surv_status_', no, '.csv'), header = FALSE, sep=','))
  
  val_feature_values <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_feature_values_', no, '.csv'), header = FALSE, sep=',')
  val_time <- as.matrix(read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_surv_time_x_', no, '.csv'), header = FALSE, sep=','))
  val_status <- as.matrix(read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_surv_status_', no, '.csv'), header = FALSE, sep=','))
  
  test_feature_values <- read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_feature_values_', no, '.csv'), header = FALSE, sep=',')
  test_time <- as.matrix(read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_surv_time_x_', no, '.csv'), header = FALSE, sep=','))
  test_status <- as.matrix(read.csv(paste0('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_surv_status_', no, '.csv'), header = FALSE, sep=','))
  
  train_time = as.vector(rbind(train_time, val_time))
  train_status = as.vector(rbind(train_status, val_status))
  train_feature_values = rbind(train_feature_values ,val_feature_values)
  
  feature_idx <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21)
  train_feature_values = train_feature_values[,feature_idx]
  test_feature_values = test_feature_values[,feature_idx]
  
  train_data = cbind(train_time, train_status, train_feature_values)
  colnames(train_data) <- c('time', 'status', colnames(train_feature_values))
  
  rf_model1 <- rfsrc(Surv(time , status) ~ ., data=train_data, ntree=1000, nodesize=6, cause=1)
  rf_model2 <- rfsrc(Surv(time , status) ~ ., data=train_data, ntree=1000, nodesize=6, cause=2)
  rf_model3 <- rfsrc(Surv(time , status) ~ ., data=train_data, ntree=1000, nodesize=6, cause=3)
  
  Test_label1 <- as.integer(test_status == 1)
  Test_label2 <- as.integer(test_status == 2)
  Test_label3 <- as.integer(test_status == 3)
  
  preds1 = predict(rf_model1, test_feature_values)$cif
  patient_no = nrow(test_feature_values)
  total_pair = 0.0
  cor_pair = 0.0
  for(i in 1:patient_no){
    if(Test_label1[i] == 1){
      T1 = test_time[i]+1
      R1 = preds1[i,T1,1]
      for(j in 1:patient_no){
        T2 = test_time[j]+1
        R2 = preds1[j,T1,1]
        if(T1 < T2){
          total_pair = total_pair+1.
          if(R1 > R2){cor_pair = cor_pair+1.}}}}}
  c_td1[no+1] = cor_pair/total_pair

  preds2 = predict(rf_model2, test_feature_values)$cif
  total_pair = 0.0
  cor_pair = 0.0
  for(i in 1:patient_no){
    if(Test_label2[i] == 1){
      T1 = test_time[i]+1
      R1 = preds2[i,T1,2]
      for(j in 1:patient_no){
        T2 = test_time[j]+1
        R2 = preds2[j,T1,2]
        if(T1 < T2){
          total_pair = total_pair+1.
          if(R1 > R2){cor_pair = cor_pair+1.}}}}}
  c_td2[no+1] = cor_pair/total_pair

  preds3 = predict(rf_model3, test_feature_values)$cif
  total_pair = 0.0
  cor_pair = 0.0
  for(i in 1:patient_no){
    if(Test_label3[i] == 1){
      T1 = test_time[i]+1
      R1 = preds3[i,T1,3]
      for(j in 1:patient_no){
        T2 = test_time[j]+1
        R2 = preds3[j,T1,3]
        if(T1 < T2){
          total_pair = total_pair+1.
          if(R1 > R2){cor_pair = cor_pair+1.}}}}}
  c_td3[no+1] = cor_pair/total_pair

}
c_td1 = unlist(c_td1)

print(mean(c_td1))
print(mean(c_td1)+sd(c_td1)*1.96)
print(mean(c_td1)-sd(c_td1)*1.96)

c_td2 = unlist(c_td2)

print(mean(c_td2))
print(mean(c_td2)+sd(c_td2)*1.96)
print(mean(c_td2)-sd(c_td2)*1.96)

c_td3 = unlist(c_td3)

print(mean(c_td3))
print(mean(c_td3)+sd(c_td3)*1.96)
print(mean(c_td3)-sd(c_td3)*1.96)
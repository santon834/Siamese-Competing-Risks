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
   
  FG_survival1 = matrix(-1, nrow=nrow(test_feature_values), ncol=30)
  FG_survival2 = matrix(-1, nrow=nrow(test_feature_values), ncol=30)
  FG_survival3 = matrix(-1, nrow=nrow(test_feature_values), ncol=30)
  
  feature_idx <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21)
  FGmodel1 <- crr(train_time, train_status, train_feature_values[,feature_idx], failcode=1, cencode=0)
  for(j in 1:nrow(test_feature_values)){
    tmp_pred <- predict(FGmodel1, test_feature_values[j,feature_idx]) 
    FG_survival1[j,] = t(tmp_pred[,2])
    
  }
  Test_label1 <- as.integer(test_status == 1)
  FGmodel2 <- crr(train_time, train_status, train_feature_values[,feature_idx], failcode=2, cencode=0)
  for(j in 1:nrow(test_feature_values)){
    tmp_pred <- predict(FGmodel2, test_feature_values[j,feature_idx]) 
    FG_survival2[j,] = t(tmp_pred[,2])
    
  }
  Test_label2 <- as.integer(test_status == 2)
  
  FGmodel3 <- crr(train_time, train_status, train_feature_values[,feature_idx], failcode=3, cencode=0)
  for(j in 1:nrow(test_feature_values)){
    tmp_pred <- predict(FGmodel3, test_feature_values[j,feature_idx]) 
    FG_survival3[j,] = t(tmp_pred[,2])
    
  }
  Test_label3 <- as.integer(test_status == 3)

  patient_no = nrow(test_feature_values)
  total_pair = 0.0
  cor_pair = 0.0
  for(i in 1:patient_no){
    if(Test_label1[i] == 1){
      T1 = test_time[i]+1
      R1 = FG_survival1[i,T1]
      for(j in 1:patient_no){
        T2 = test_time[j]+1
        R2 = FG_survival1[j,T1]
        if(T1 < T2){
          total_pair = total_pair+1.
          if(R1 > R2){cor_pair = cor_pair+1.}}}}}
  c_td1[no+1] = cor_pair/total_pair

  total_pair = 0.0
  cor_pair = 0.0
  for(i in 1:patient_no){
    if(Test_label2[i] == 1){
      T1 = test_time[i]+1
      R1 = FG_survival2[i,T1]
      for(j in 1:patient_no){
        T2 = test_time[j]+1
        R2 = FG_survival2[j,T1]
        if(T1 < T2){
          total_pair = total_pair+1.
          if(R1 > R2){cor_pair = cor_pair+1.}}}}}
  c_td2[no+1] = cor_pair/total_pair

  total_pair = 0.0
  cor_pair = 0.0
  for(i in 1:patient_no){
    if(Test_label3[i] == 1){
      T1 = test_time[i]+1
      R1 = FG_survival3[i,T1]
      for(j in 1:patient_no){
        T2 = test_time[j]+1
        R2 = FG_survival3[j,T1]
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
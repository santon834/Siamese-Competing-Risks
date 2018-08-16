# imports
import numpy as np
import deepsurv

## run deepsurv ## 
# typ: event type
def ds(typ):

	c_idx = np.zeros(5)	

	# for each cross validation set
	for no in range(5):
		train_features_c = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/train_c_feature_values_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float32)
		train_features_uc = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/train_uc_feature_values_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float32)
		train_features = np.append(train_features_c, train_features_uc, axis=0)
		train_labels = np.append(np.zeros(train_features_c.shape[0], dtype=np.int32), np.ones(train_features_uc.shape[0], dtype=np.int32), axis=0)
		train_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/train_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',')

		val_features_c = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/val_c_feature_values_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float32)
		val_features_uc = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/val_uc_feature_values_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float32)
		val_features = np.append(val_features_c, val_features_uc, axis=0)
		val_labels = np.append(np.zeros(val_features_c.shape[0], dtype=np.int32), np.ones(val_features_uc.shape[0], dtype=np.int32), axis=0)
		val_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/val_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',')

		test_features_c = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_c_feature_values_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float32)
		test_features_uc = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_uc_feature_values_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float32)
		test_features = np.append(test_features_c, test_features_uc, axis=0)
		test_labels = np.append(np.zeros(test_features_c.shape[0], dtype=np.int32), np.ones(test_features_uc.shape[0], dtype=np.int32), axis=0)
		test_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',')

		train_data = {'x': train_features, 't': train_surv_values, 'e': train_labels}
		val_data = {'x': val_features, 't': val_surv_values, 'e': val_labels}
		test_data = {'x': test_features, 't': test_surv_values, 'e': test_labels}

		# hyper parameters
		n_in = train_features.shape[1]
		lr = 0.0000000001
		l = np.rint((train_features.shape[1]+np.amax(train_surv_values))/2).astype(int)
		n_hidden = [l,l]
		dp = 0.3
		bn = True
		l2_reg = 0
		l1_reg = 0
	
		# run 
		network = deepsurv.DeepSurv(n_in=n_in, learning_rate=lr, hidden_layers_sizes=n_hidden, dropout=dp, batch_norm=bn, L2_reg = l2_reg, L1_reg = l1_reg)
		log = network.train(train_data, val_data, n_epochs=1000) #500
		
		# get results
		preds = network.predict_risk(test_features)		
		patient_no = test_features.shape[0]

		total_pair = 0.0
    		cor_pair = 0.0
    	
    		for i in range(patient_no):
        		if(test_labels[i] == 1):
            			T1 = test_surv_values[i]
           			R1 = preds[i]
            			for j in range(patient_no):
                   			T2 = test_surv_values[j]
                    			R2 = preds[j]
                    			if(T1 < T2):
                        			total_pair += 1.
                        			if(R1 > R2):
                            				cor_pair += 1.
    
    		c_idx[no] = cor_pair/total_pair
    		print(c_idx[no])
			
	print(np.mean(c_idx))
	print(np.mean(c_idx)+np.std(c_idx)*1.96)
	print(np.mean(c_idx)-np.std(c_idx)*1.96)

ds(1)
ds(2)
ds(3)
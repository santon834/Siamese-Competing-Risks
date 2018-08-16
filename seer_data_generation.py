# imports
from sklearn.preprocessing import StandardScaler
import numpy as np

## process the raw data ##
# val_size: % of trainset, test_size: % of dataset, interval_no: # of label intervals,
# cval_no: cross validation set, typ: event type, dual: algorithm type (cs events*3 or the original 3)
def get_uniform_data(val_size, test_size, interval_no, cval_no, typ, dual): 

	data = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER.csv', delimiter=',', skip_header=1)

	# the type of labels required for the algorithm
	if(dual == 0):
		if(typ == 1): 
			data[data[:,1] == 2,1] = 0
			data[data[:,1] == 3,1] = 0
		if(typ == 2): 
			data[data[:,1] == 1,1] = 0	
			data[data[:,1] == 2,1] = 1
			data[data[:,1] == 3,1] = 0
		if(typ == 3): 
			data[data[:,1] == 1,1] = 0	
			data[data[:,1] == 2,1] = 0
			data[data[:,1] == 3,1] = 1		

	# randomize dataset
	rndm_idx = np.random.permutation(range(data.shape[0]))
	data = data[rndm_idx,:]
	surv_values = data[:,:2].astype(int)
	feature_values = data[:,2:].astype(np.float16)

	# divide to cencored and uncencored
	c_feature_values = feature_values[surv_values[:,1] == 0,:]
	c_surv_values = surv_values[surv_values[:,1] == 0,:]
	c_patient_no = c_surv_values.shape[0]

	if(dual == 0):
		uc_feature_values = feature_values[surv_values[:,1] == 1,:]
		uc_surv_values = surv_values[surv_values[:,1] == 1,:]
		uc_patient_no = uc_surv_values.shape[0]

	else:
		uc_feature_values = feature_values[surv_values[:,1] != 0,:]
		uc_surv_values = surv_values[surv_values[:,1] != 0,:]
		uc_patient_no = uc_surv_values.shape[0]

	# divide to train, val and test sets
	c_train_idx = (np.rint((1.-test_size)*c_patient_no)).astype(int)
	c_val_idx = (np.rint(val_size*c_train_idx)).astype(int)

	uc_train_idx = (np.rint((1.-test_size)*uc_patient_no)).astype(int)
	uc_val_idx = (np.rint(val_size*uc_train_idx)).astype(int)

	train_c_feature_values = c_feature_values[:c_train_idx-c_val_idx,:]
	train_c_surv_values = c_surv_values[:c_train_idx-c_val_idx,:]
	train_c_patient_no = c_train_idx-c_val_idx

	train_uc_feature_values = uc_feature_values[:uc_train_idx-uc_val_idx,:]
	train_uc_surv_values = uc_surv_values[:uc_train_idx-uc_val_idx,:]
	train_uc_patient_no = uc_train_idx-uc_val_idx

	train_feature_values = np.append(train_c_feature_values, train_uc_feature_values, axis=0)
	train_surv_values = np.append(train_c_surv_values, train_uc_surv_values, axis=0)
	train_patient_no = train_c_patient_no + train_uc_patient_no

	val_c_feature_values = c_feature_values[c_train_idx-c_val_idx:c_train_idx,:]
	val_c_surv_values = c_surv_values[c_train_idx-c_val_idx:c_train_idx,:]
	val_c_patient_no = c_val_idx

	val_uc_feature_values = uc_feature_values[uc_train_idx-uc_val_idx:uc_train_idx,:]
	val_uc_surv_values = uc_surv_values[uc_train_idx-uc_val_idx:uc_train_idx,:]
	val_uc_patient_no = uc_val_idx

	val_feature_values = np.append(val_c_feature_values, val_uc_feature_values, axis=0)
	val_surv_values = np.append(val_c_surv_values, val_uc_surv_values, axis=0)
	val_patient_no = val_c_patient_no + val_uc_patient_no

	test_c_feature_values = c_feature_values[c_train_idx:,:]
	test_c_surv_values = c_surv_values[c_train_idx:,:]
	test_c_patient_no = c_patient_no - c_train_idx

	test_uc_feature_values = uc_feature_values[uc_train_idx:,:]
	test_uc_surv_values = uc_surv_values[uc_train_idx:,:]
	test_uc_patient_no = uc_patient_no - uc_train_idx

	test_feature_values = np.append(test_c_feature_values, test_uc_feature_values, axis=0)
	test_surv_values = np.append(test_c_surv_values, test_uc_surv_values, axis=0)
	test_patient_no = test_c_patient_no + test_uc_patient_no

	# standardize features
	std_handler = StandardScaler().fit(train_feature_values)
	std_handler.transform(train_feature_values, copy=False)
	std_handler.transform(train_c_feature_values, copy=False)
	std_handler.transform(train_uc_feature_values, copy=False)
	std_handler.transform(val_feature_values, copy=False)
	std_handler.transform(val_c_feature_values, copy=False)
	std_handler.transform(val_uc_feature_values, copy=False)
	std_handler.transform(test_feature_values, copy=False)
	std_handler.transform(test_c_feature_values, copy=False)
	std_handler.transform(test_uc_feature_values, copy=False)

	# convert labels to one-hot over intervals
	interval_values = np.zeros(interval_no+1, dtype=int)
	label_dictionary = np.zeros(np.amax(surv_values[:,0])+1, dtype=int)

	train_c_labels = np.zeros((train_c_patient_no, interval_no), dtype=int)
	train_uc_labels = np.zeros((train_uc_patient_no, interval_no), dtype=int)
	train_labels = np.zeros((train_patient_no, interval_no), dtype=int)

	val_c_labels = np.zeros((val_c_patient_no, interval_no), dtype=int)
	val_uc_labels = np.zeros((val_uc_patient_no, interval_no), dtype=int)
	val_labels = np.zeros((val_patient_no, interval_no), dtype=int)

	test_c_labels = np.zeros((test_c_patient_no, interval_no), dtype=int)
	test_uc_labels = np.zeros((test_uc_patient_no, interval_no), dtype=int)
	test_labels = np.zeros((test_patient_no, interval_no), dtype=int)

	train_c_times = np.zeros(train_c_patient_no, dtype=int)
	train_uc_times = np.zeros(train_uc_patient_no, dtype=int)
	train_times = np.zeros(train_patient_no, dtype=int)

	val_c_times = np.zeros(val_c_patient_no, dtype=int)
	val_uc_times = np.zeros(val_uc_patient_no, dtype=int)
	val_times = np.zeros(val_patient_no, dtype=int)

	test_c_times = np.zeros(test_c_patient_no, dtype=int)
	test_uc_times = np.zeros(test_uc_patient_no, dtype=int)
	test_times = np.zeros(test_patient_no, dtype=int)	

	sorted_surv_times = np.sort(train_uc_surv_values[:,0])

	interval_values = np.zeros(interval_no+1)
	for i in range(1,interval_no+1):
		interval_values[i] = sorted_surv_times[(train_uc_patient_no/interval_no)*i]
	interval_values = interval_values.astype(int).tolist()
	interval_values[interval_no] = 180
		
	for i in range(1,interval_no+1):
		label_dictionary[range(interval_values[i-1], interval_values[i])] = i-1
	
	for i in range(train_c_patient_no):
		train_c_labels[i,label_dictionary[train_c_surv_values[i,0]]] = 1
		train_c_times[i] = label_dictionary[train_c_surv_values[i,0]]

	for i in range(train_uc_patient_no):
		train_uc_labels[i,label_dictionary[train_uc_surv_values[i,0]]] = 1
		train_uc_times[i] = label_dictionary[train_uc_surv_values[i,0]]

	for i in range(train_patient_no):
		train_labels[i,label_dictionary[train_surv_values[i,0]]] = 1
		train_times[i] = label_dictionary[train_surv_values[i,0]]	

	for i in range(val_c_patient_no):
		val_c_labels[i,label_dictionary[val_c_surv_values[i,0]]] = 1
		val_c_times[i] = label_dictionary[val_c_surv_values[i,0]] 

	for i in range(val_uc_patient_no):
		val_uc_labels[i,label_dictionary[val_uc_surv_values[i,0]]] = 1
		val_uc_times[i] = label_dictionary[val_uc_surv_values[i,0]] 

	for i in range(val_patient_no):
		val_labels[i,label_dictionary[val_surv_values[i,0]]] = 1
		val_times[i] = label_dictionary[val_surv_values[i,0]] 

	for i in range(test_c_patient_no):
		test_c_labels[i,label_dictionary[test_c_surv_values[i,0]]] = 1 
		test_c_times[i] = label_dictionary[test_c_surv_values[i,0]] 

	for i in range(test_uc_patient_no):
		test_uc_labels[i,label_dictionary[test_uc_surv_values[i,0]]] = 1
		test_uc_times[i] = label_dictionary[test_uc_surv_values[i,0]] 

	for i in range(test_patient_no):
		test_labels[i,label_dictionary[test_surv_values[i,0]]] = 1
		test_times[i] = label_dictionary[test_surv_values[i,0]] 			

	# save
	if(dual == 0):
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_c_labels_{}_{}.csv'.format(cval_no,typ), train_c_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_uc_labels_{}_{}.csv'.format(cval_no,typ), train_uc_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_labels_{}_{}.csv'.format(cval_no,typ), train_labels, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_c_labels_{}_{}.csv'.format(cval_no,typ), val_c_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_uc_labels_{}_{}.csv'.format(cval_no,typ), val_uc_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_labels_{}_{}.csv'.format(cval_no,typ), val_labels, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_c_labels_{}_{}.csv'.format(cval_no,typ), test_c_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_uc_labels_{}_{}.csv'.format(cval_no,typ), test_uc_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_labels_{}_{}.csv'.format(cval_no,typ), test_labels, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_c_surv_time_{}_{}.csv'.format(cval_no,typ), train_c_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_uc_surv_time_{}_{}.csv'.format(cval_no,typ), train_uc_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_surv_time_{}_{}.csv'.format(cval_no,typ), train_surv_values[:,0], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_c_surv_time_{}_{}.csv'.format(cval_no,typ), val_c_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_uc_surv_time_{}_{}.csv'.format(cval_no,typ), val_uc_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_surv_time_{}_{}.csv'.format(cval_no,typ), val_surv_values[:,0], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_c_surv_time_{}_{}.csv'.format(cval_no,typ), test_c_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_uc_surv_time_{}_{}.csv'.format(cval_no,typ), test_uc_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_surv_time_{}_{}.csv'.format(cval_no,typ), test_surv_values[:,0], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_c_surv_time_x_{}_{}.csv'.format(cval_no,typ), train_c_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_uc_surv_time_x_{}_{}.csv'.format(cval_no,typ), train_uc_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_surv_time_x_{}_{}.csv'.format(cval_no,typ), train_times, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_c_surv_time_x_{}_{}.csv'.format(cval_no,typ), val_c_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_uc_surv_time_x_{}_{}.csv'.format(cval_no,typ), val_uc_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_surv_time_x_{}_{}.csv'.format(cval_no,typ), val_times, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_c_surv_time_x_{}_{}.csv'.format(cval_no,typ), test_c_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_uc_surv_time_x_{}_{}.csv'.format(cval_no,typ), test_uc_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_surv_time_x_{}_{}.csv'.format(cval_no,typ), test_times, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_c_feature_values_{}_{}.csv'.format(cval_no,typ), train_c_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_uc_feature_values_{}_{}.csv'.format(cval_no,typ), train_uc_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/train_feature_values_{}_{}.csv'.format(cval_no,typ), train_feature_values, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_c_feature_values_{}_{}.csv'.format(cval_no,typ), val_c_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_uc_feature_values_{}_{}.csv'.format(cval_no,typ), val_uc_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/val_feature_values_{}_{}.csv'.format(cval_no,typ), val_feature_values, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_c_feature_values_{}_{}.csv'.format(cval_no,typ), test_c_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_uc_feature_values_{}_{}.csv'.format(cval_no,typ), test_uc_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER/test_feature_values_{}_{}.csv'.format(cval_no,typ), test_feature_values, delimiter=",")

	else:
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_c_labels_{}.csv'.format(cval_no), train_c_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_uc_labels_{}.csv'.format(cval_no), train_uc_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_labels_{}.csv'.format(cval_no), train_labels, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_c_labels_{}.csv'.format(cval_no), val_c_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_uc_labels_{}.csv'.format(cval_no), val_uc_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_labels_{}.csv'.format(cval_no), val_labels, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_c_labels_{}.csv'.format(cval_no), test_c_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_uc_labels_{}.csv'.format(cval_no), test_uc_labels, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_labels_{}.csv'.format(cval_no), test_labels, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_c_surv_time_{}.csv'.format(cval_no), train_c_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_uc_surv_time_{}.csv'.format(cval_no), train_uc_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_surv_time_{}.csv'.format(cval_no), train_surv_values[:,0], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_c_surv_time_{}.csv'.format(cval_no), val_c_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_uc_surv_time_{}.csv'.format(cval_no), val_uc_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_surv_time_{}.csv'.format(cval_no), val_surv_values[:,0], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_c_surv_time_{}.csv'.format(cval_no), test_c_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_uc_surv_time_{}.csv'.format(cval_no), test_uc_surv_values[:,0], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_surv_time_{}.csv'.format(cval_no), test_surv_values[:,0], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_c_surv_status_{}.csv'.format(cval_no), train_c_surv_values[:,1], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_uc_surv_status_{}.csv'.format(cval_no), train_uc_surv_values[:,1], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_surv_status_{}.csv'.format(cval_no), train_surv_values[:,1], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_c_surv_status_{}.csv'.format(cval_no), val_c_surv_values[:,1], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_uc_surv_status_{}.csv'.format(cval_no), val_uc_surv_values[:,1], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_surv_status_{}.csv'.format(cval_no), val_surv_values[:,1], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_c_surv_status_{}.csv'.format(cval_no), test_c_surv_values[:,1], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_uc_surv_status_{}.csv'.format(cval_no), test_uc_surv_values[:,1], delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_surv_status_{}.csv'.format(cval_no), test_surv_values[:,1], delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_c_surv_time_x_{}.csv'.format(cval_no), train_c_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_uc_surv_time_x_{}.csv'.format(cval_no), train_uc_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_surv_time_x_{}.csv'.format(cval_no), train_times, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_c_surv_time_x_{}.csv'.format(cval_no), val_c_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_uc_surv_time_x_{}.csv'.format(cval_no), val_uc_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_surv_time_x_{}.csv'.format(cval_no), val_times, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_c_surv_time_x_{}.csv'.format(cval_no), test_c_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_uc_surv_time_x_{}.csv'.format(cval_no), test_uc_times, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_surv_time_x_{}.csv'.format(cval_no), test_times, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_c_feature_values_{}.csv'.format(cval_no), train_c_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_uc_feature_values_{}.csv'.format(cval_no), train_uc_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_feature_values_{}.csv'.format(cval_no), train_feature_values, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_c_feature_values_{}.csv'.format(cval_no), val_c_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_uc_feature_values_{}.csv'.format(cval_no), val_uc_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_feature_values_{}.csv'.format(cval_no), val_feature_values, delimiter=",")

		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_c_feature_values_{}.csv'.format(cval_no), test_c_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_uc_feature_values_{}.csv'.format(cval_no), test_uc_feature_values, delimiter=",")
		np.savetxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_feature_values_{}.csv'.format(cval_no), test_feature_values, delimiter=",")

	return 

for i in range(5):
	get_uniform_data(val_size=0.10, test_size=0.20, interval_no=30, cval_no=i, typ=1, dual=0)
	get_uniform_data(val_size=0.10, test_size=0.20, interval_no=30, cval_no=i, typ=2, dual=0)
	get_uniform_data(val_size=0.10, test_size=0.20, interval_no=30, cval_no=i, typ=3, dual=0)
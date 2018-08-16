# imports
import numpy as np
import tensorflow as tf
import sys
import selu as se
import matplotlib.pyplot as plt
 
def cost_function(predictions_left, predictions_right, labels_left, labels_right, w):
	alpha = 1. #1000000.
	diff = -tf.log(tf.sigmoid(500*(predictions_left - predictions_right))+1e-10)
	ctd = tf.multiply(diff, labels_left)

	# diff2 = tf.sigmoid(10*(predictions_left - predictions_right))
	# diff3 = tf.sigmoid(10*(predictions_right - predictions_left))
	# left_valid = 1.-tf.cumsum(labels_left, axis=1)
	# right_valid = 1.-tf.cumsum(labels_right, axis=1)

	# reg_left = 10.*tf.square(tf.multiply(diff2, tf.multiply(1./tf.cumsum(left_valid+labels_left, axis=1), left_valid)))
	# reg_left_2 = 0.1*tf.multiply(diff3, right_valid-left_valid-labels_left)

	# exp = 1./tf.cumsum(tf.cumsum(labels_left, axis=1), axis=1)
	# exp2 = tf.where(tf.is_nan(exp), exp, tf.zeros_like(labels_left))
	# reg_left_2 = 10.*tf.multiply(diff2, exp2)
	# reg_left_3 = 10*tf.multiply(diff2, 1.-right_valid)
	# reg_right = 1000.*tf.square(tf.multiply(predictions_right, tf.multiply(1./tf.cumsum(right_valid, axis=1), right_valid)))

	# return tf.reduce_sum(tf.multiply(tf.reduce_mean(alpha*ctd + reg_left + reg_right+reg_left_2+reg_left_3, axis=0), w))
	return tf.reduce_sum(tf.multiply(tf.reduce_mean(alpha*ctd, axis=0), w))
  
def s(data, dp, l_out, l1 ,l2, dp_type):
	# data = tf.nn.dropout(data, dp)
	# data = tf.expand_dims(data, axis=2)
	# data = tf.contrib.layers.batch_norm(data,  activation_fn=tf.nn.relu)
	# data = tf.contrib.layers.conv2d(data, 1, 3, 1, 'SAME', activation_fn=None)
	# data = tf.contrib.layers.max_pool2d(data, 3, 2, 'SAME')
	if(dp_type == True):
		# l1 = tf.contrib.layers.batch_norm(data,  activation_fn=tf.nn.relu, is_training=bn)
		layer1 = tf.contrib.layers.fully_connected(data, num_outputs=l_out, activation_fn=se.selu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=l1, scale_l2=l2), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
		# l1 = tf.concat([data, l1], axis=1)
		layer1 = se.dropout_selu(layer1, rate=dp, training=(dp < 1.))
		# l2 = tf.contrib.layers.batch_norm(l1,  activation_fn=tf.nn.relu, is_training=bn)
		layer2 = tf.contrib.layers.fully_connected(layer1, num_outputs=l_out, activation_fn=se.selu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=l1, scale_l2=l2), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
		# l2 = tf.concat([l1, l2], axis=1)
		layer2 = se.dropout_selu(layer2, rate=dp, training=(dp < 1.))
		# l3 = tf.contrib.layers.batch_norm(l2,  activation_fn=tf.nn.relu, is_training=bn)
		layer3 = tf.contrib.layers.fully_connected(layer2, num_outputs=l_out, activation_fn=se.selu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=l1, scale_l2=l2), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
		
	else:
		# l1 = tf.contrib.layers.batch_norm(data,  activation_fn=tf.nn.relu, is_training=bn)
		layer1 = tf.contrib.layers.fully_connected(data, num_outputs=l_out, activation_fn=se.selu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=l1, scale_l2=l2), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
		# l1 = tf.concat([data, l1], axis=1)
		# l2 = tf.contrib.layers.batch_norm(l1,  activation_fn=tf.nn.relu, is_training=bn)
		layer2 = tf.contrib.layers.fully_connected(tf.nn.dropout(layer1, dp), num_outputs=l_out, activation_fn=se.selu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=l1, scale_l2=l2), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
		# l2 = tf.concat([l1, l2], axis=1)
		# l3 = tf.contrib.layers.batch_norm(l2,  activation_fn=tf.nn.relu, is_training=bn)
		layer3 = tf.contrib.layers.fully_connected(tf.nn.dropout(layer2, dp), num_outputs=l_out, activation_fn=se.selu, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=l1, scale_l2=l2), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))

	return layer3	

def h(data, dp, h):
	h10 = tf.contrib.layers.fully_connected(inputs=data, num_outputs=h, activation_fn=None, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0., scale_l2=0.), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
	h10 = tf.nn.softmax(h10)
	# h10 = tf.divide(h10, tf.tile(tf.reduce_sum(h10, axis=1, keep_dims=True), [1, h]) + 1e-10)
	h10 = tf.cumsum(h10, axis=1)

	# layer_out = tf.nn.dropout(layer_in, dp)
	return h10	
	
def c_index2(predictions_left, predictions_right, fake_labels):
	more = tf.reduce_sum(tf.multiply(tf.to_float(predictions_left > predictions_right), fake_labels), axis=1)
	return tf.reduce_sum(more)

def brier_score(predictions_left, labels_left, fake_labels):
	return tf.reduce_mean(tf.multiply(tf.square(tf.cumsum(labels_left, axis=1) - predictions_left), fake_labels))
 
def brier_score_np(predictions_left, labels_left, fake_labels):
	return np.mean(np.multiply(np.square(np.cumsum(labels_left, axis=1) - predictions_left), fake_labels))	

def train_algorithm(no, train_pairs, train_features, train_labels, val_pairs, val_features, val_labels, steps, weights, shuffle, batch_size, epoch_no, learning_rate, lr_decay, l1, l2, max_grad, l_out, val_step, dp_rate, dp_type):
	val_batch_size = 1024 #32768
	features_no = train_features.shape[1]
	train_no = train_pairs.shape[0]
	val_no = val_pairs.shape[0]
	total_batch = train_no/batch_size
	val_parts = val_no/val_batch_size
	total_val = (total_batch/val_step)+1

	logs_path = '/home/anton/Documents/anton/ICLR' 

	tf.reset_default_graph()

	dp = tf.placeholder(tf.float32, [])
	lr = tf.placeholder(tf.float32, [])
	w = tf.placeholder(tf.float32, [steps])
	features_left = tf.placeholder(tf.float32, [None, features_no])
	features_right = tf.placeholder(tf.float32, [None, features_no])
	labels_left = tf.placeholder(tf.float32, [None, steps])
	labels_right = tf.placeholder(tf.float32, [None, steps])
	s_shared = tf.make_template('s_shared', s)
	h_shared = tf.make_template('h_shared', h)

	states_left = s_shared(features_left, dp, l_out, l1, l2, dp_type)
	states_right = s_shared(features_right, dp, l_out, l1, l2, dp_type)
	predictions_left = h_shared(states_left, dp, steps)
	predictions_right = h_shared(states_right, dp, steps)
	pl = tf.reduce_mean(tf.reduce_sum(predictions_left, axis=1))
	pr = tf.reduce_mean(tf.reduce_sum(predictions_right, axis=1))

	cost = cost_function(predictions_left, predictions_right, labels_left, labels_right, w)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	trainer = optimizer.minimize(cost)

	ctd_index2 = c_index2(predictions_left, predictions_right, labels_left)
	br_score = brier_score(predictions_left, labels_left, labels_left)

	init = tf.global_variables_initializer()
	init2 = tf.local_variables_initializer()
	merged_summary_op_train = tf.summary.merge([tf.summary.scalar("Train Loss", cost), tf.summary.scalar("Train Ctd Index", ctd_index2), tf.summary.scalar("Train Left Mean", pl), tf.summary.scalar("Train Right Mean", pr)])
	merged_summary_op_val = tf.summary.merge([tf.summary.scalar("Val Ctd Index", ctd_index2), tf.summary.scalar("Val Left Mean", pl), tf.summary.scalar("Val Right Mean", pr)])
	best_val = -np.inf
	with tf.Session() as sess:
		saver = tf.train.Saver(max_to_keep=1)
		sess.run(init)
		sess.run(init2)
		summary_writer_train = tf.summary.FileWriter(logs_path + '/logs/train', graph=tf.get_default_graph())
		summary_writer_val = tf.summary.FileWriter(logs_path + '/logs/val', graph=tf.get_default_graph())
		for epoch in range(epoch_no):
			if(shuffle == True):
				rndm_idx = np.random.permutation(range(train_no))
				train_pairs = train_pairs[rndm_idx,:]
			avg_cost = 0.
			avg_ctd = 0.
			for i in range(total_batch):
				rndm_idx = np.random.randint(0, high=train_no, size=(batch_size))
				batch_left = train_features[train_pairs[rndm_idx,0],:]
				batch_right = train_features[train_pairs[rndm_idx,1],:]
				l_left = train_labels[train_pairs[rndm_idx,0],:]
				l_right = train_labels[train_pairs[rndm_idx,1],:]
				_, a, b, c, summary = sess.run([trainer, cost, ctd_index2, br_score, merged_summary_op_train], feed_dict={lr: learning_rate, dp: dp_rate, features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right, w: weights})
				summary_writer_train.add_summary(summary, epoch * total_batch + i)
				avg_cost += a / total_batch
				avg_ctd += b / total_batch
				sys.stdout.write("--->>> Batch: {}/{}, Loss: {}, Ctd Index: {} , Brier Score: {} <<<---\r".format(i, total_batch, a/batch_size, b/batch_size, c))
				sys.stdout.flush()
				
				# validation
				if((i%val_step == 0) & (i > 0)):
					sess.run(init2)
					tot_val = 0
					tot_b = 0
					for k in range(val_parts):
						batch_left = val_features[val_pairs[k*val_batch_size:(k+1)*val_batch_size,0],:]
						batch_right = val_features[val_pairs[k*val_batch_size:(k+1)*val_batch_size,1],:]
						l_left = val_labels[val_pairs[k*val_batch_size:(k+1)*val_batch_size,0],:]
						l_right = val_labels[val_pairs[k*val_batch_size:(k+1)*val_batch_size,1],:]		
						val_a, val_b, val_summary= sess.run([ctd_index2, br_score, merged_summary_op_val], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
						tot_val += val_a
						
						tot_b += val_b*val_batch_size
											
					k += 1
					batch_left = val_features[val_pairs[k*val_batch_size:val_no,0],:]
					batch_right = val_features[val_pairs[k*val_batch_size:val_no,1],:]
					l_left = val_labels[val_pairs[k*val_batch_size:val_no,0],:]
					l_right = val_labels[val_pairs[k*val_batch_size:val_no,1],:]		
					val_a, val_b, val_summary = sess.run([ctd_index2, br_score, merged_summary_op_val], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
					tot_val += val_a
					
					tot_b += val_b*(val_no-k*val_batch_size)
											
					print 	
					print("--->>> Inner Val Ctd Index: {}, Val Brier Score {}".format(tot_val/val_no, tot_b/val_no))
					if(tot_val/val_no > best_val):
						print("Saving model...")
						saver.save(sess, logs_path + '/models/model', global_step=epoch)
						best_val = tot_val/val_no

			print 
			print("--->>> Epoch: {}/{}".format(epoch+1,epoch_no), ", Loss: ", "{:.9f}".format(avg_cost), "Ctd Index: ", "{:.9f} <<<---".format(avg_ctd))	
 
			# validation
			tot_val = 0
			tot_b = 0
			sess.run(init2)
			for k in range(val_parts):
				batch_left = val_features[val_pairs[k*val_batch_size:(k+1)*val_batch_size,0],:]
				batch_right = val_features[val_pairs[k*val_batch_size:(k+1)*val_batch_size,1],:]
				l_left = val_labels[val_pairs[k*val_batch_size:(k+1)*val_batch_size,0],:]
				l_right = val_labels[val_pairs[k*val_batch_size:(k+1)*val_batch_size,1],:]		
				val_a, val_b, val_summary = sess.run([ctd_index2, br_score, merged_summary_op_val], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
				tot_val += val_a
				tot_b += val_b*val_batch_size
							
			k += 1
			batch_left = val_features[val_pairs[k*val_batch_size:val_no,0],:]
			batch_right = val_features[val_pairs[k*val_batch_size:val_no,1],:]
			l_left = val_labels[val_pairs[k*val_batch_size:val_no,0],:]
			l_right = val_labels[val_pairs[k*val_batch_size:val_no,1],:]		
			val_a, val_b, val_summary = sess.run([ctd_index2, br_score, merged_summary_op_val], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
			tot_val += val_a
			tot_b += val_b*(val_no-k*val_batch_size)
							
			print 	
			print("--->>> Outer Val Ctd Index: {}, Val Brier Score {}".format(tot_val/val_no, tot_b/val_no))
			if(tot_val/val_no > best_val):
				print("Saving model...")
				saver.save(sess, logs_path + '/models/model_{}'.format(no), global_step=epoch)
				best_val = tot_val/val_no	

def test_algorithm(train_pairs, train_features, train_labels, steps, epoch_no, learning_rate, l1, l2, dp_type, l_out, fake_labels):
	max_a = -1
	features_no = train_features.shape[1]
	train_no = train_pairs.shape[0]
	bt = 262144

	logs_path = '/home/anton/Documents/anton/ICLR' 

	tf.reset_default_graph()

	dp = tf.placeholder(tf.float32, [])
	w = tf.placeholder(tf.float32, [steps])
	features_left = tf.placeholder(tf.float32, [None, features_no])
	features_right = tf.placeholder(tf.float32, [None, features_no])
	labels_left = tf.placeholder(tf.float32, [None, steps])
	labels_right = tf.placeholder(tf.float32, [None, steps])
	s_shared = tf.make_template('s_shared', s)
	h_shared = tf.make_template('h_shared', h)

	states_left = s_shared(features_left, dp, l_out, l1, l2, dp_type)
	states_right = s_shared(features_right, dp, l_out, l1, l2, dp_type)
	predictions_left = h_shared(states_left, dp, steps)
	predictions_right = h_shared(states_right, dp, steps)
	cost = cost_function(predictions_left, predictions_right, labels_left, labels_right, w)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	trainer = optimizer.minimize(cost)

	ctd_index2 = c_index2(predictions_left, predictions_right, labels_left)
	with tf.Session() as sess:
		saver = tf.train.Saver(max_to_keep=epoch_no)
		saver.restore(sess,tf.train.latest_checkpoint(logs_path + '/models/'))
		a = 0.	

		estimates = sess.run([predictions_left], feed_dict={dp: 1., features_left: train_features, features_right: train_features, labels_left: train_labels, labels_right: fake_labels})[0]
		uc_idx = np.unique(train_pairs[:,0])

		b = brier_score_np(estimates[uc_idx,:], train_labels[uc_idx,:], fake_labels[uc_idx,:])
		k=0
		for k in range(train_pairs.shape[0]/bt):
			batch_left = train_features[train_pairs[k*bt:(k+1)*bt,0],:]
			batch_right = train_features[train_pairs[k*bt:(k+1)*bt,1],:]
			l_left = fake_labels[train_pairs[k*bt:(k+1)*bt,0],:]
			l_right = fake_labels[train_pairs[k*bt:(k+1)*bt,1],:]

			[a_t] = sess.run([ctd_index2], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
			a += a_t
					
			sys.stdout.write("{}/{}\r".format(k,train_pairs.shape[0]/bt))
			sys.stdout.flush()
		k += 1
		batch_left = train_features[train_pairs[k*bt:train_pairs.shape[0],0],:]
		batch_right = train_features[train_pairs[k*bt:train_pairs.shape[0],1],:]
		l_left = fake_labels[train_pairs[k*bt:train_pairs.shape[0],0],:]
		l_right = fake_labels[train_pairs[k*bt:train_pairs.shape[0],1],:]

		a_t = sess.run([ctd_index2], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})[0]
		a += a_t

	return [a/train_no, estimates, b, train_no]

def train(typ,no, batch_size, shuffle, epoch_no, learning_rate, lr_decay, l1_reg, l2_reg, max_grad, val_step, dp_rate, dpt, l_out):
	train_features = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/train_feature_values_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float16)
	train_labels = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/train_labels_{}_{}.csv'.format(no,typ), delimiter=',').astype(int)
	train_uc_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/train_uc_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float16)
	train_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/train_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float16)
	train_uc_patient_no = train_uc_surv_values.shape[0]
	train_pairs = []
	train_pairs_cat = np.zeros(30)
	for i in range(train_uc_patient_no):
		tmp = np.where(train_uc_surv_values[i] < train_surv_values[:])[0]
		tmp = np.stack([np.ones(tmp.shape[0], dtype=int)*(i+train_surv_values.shape[0]-train_uc_patient_no), tmp], axis=1)
		train_pairs += [tmp.astype(np.int32)]
		train_pairs_cat[train_uc_surv_values[i].astype(int)] += tmp.shape[0]
	train_pairs_cat[29] = 1
	train_pairs_cat[:29] = np.divide(np.ones(29), train_pairs_cat[:29]+1)
	train_pairs_cat[:29] /= np.sum(train_pairs_cat[:29])/100.	#100
	train_pairs = np.vstack(train_pairs)

	print("Train pairs: ", train_pairs.shape[0])
	print("Train cat: {}".format(train_pairs_cat))

	val_features = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/val_feature_values_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float16)
	val_labels = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/val_labels_{}_{}.csv'.format(no,typ), delimiter=',').astype(int)

	val_uc_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/val_uc_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float16)
	val_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/val_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',', dtype=np.float16)

	val_uc_patient_no = val_uc_surv_values.shape[0]
	val_pairs = []
	for i in range(val_uc_patient_no):
		tmp = np.where(val_uc_surv_values[i] < val_surv_values[:])[0]
		tmp = np.stack([np.ones(tmp.shape[0], dtype=int)*(i+val_surv_values.shape[0]-val_uc_patient_no), tmp], axis=1)
		val_pairs += [tmp.astype(np.int32)]

	val_pairs = np.vstack(val_pairs)

	print("Val pairs: ", val_pairs.shape[0])	

	train_algorithm(no, train_pairs, train_features, train_labels, val_pairs, val_features, val_labels, 30, train_pairs_cat, shuffle, batch_size, epoch_no, learning_rate, lr_decay, l1_reg, l2_reg, max_grad, l_out, val_step, dp_rate, dpt)

def test(typ, no, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out):
	test_uc_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_uc_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',')

	test_features = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_feature_values_{}_{}.csv'.format(no,typ), delimiter=',')
	test_labels = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_labels_{}_{}.csv'.format(no,typ), delimiter=',').astype(int)
	test_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_surv_time_x_{}_{}.csv'.format(no,typ), delimiter=',')

	test_uc_patient_no = test_uc_surv_values.shape[0]
	test_c_patient_no = test_surv_values.shape[0]-test_uc_patient_no

	W = np.zeros(30)
	AUC = np.zeros(30)
	BR = np.zeros(30)
	[Ctd_full, estimates, _, _] = test_t(test_uc_surv_values, 30, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out, test_features, test_labels, test_surv_values, test_labels)
	print("Ctd: {}".format(Ctd_full))
	np.savetxt('/home/anton/Documents/anton/ICLR/results/SEER/test_full_estimates_x_{}_{}.csv'.format(no,typ), estimates, delimiter=",")
	patient_no = estimates.shape[0]
	total_pair = 0.0
    	cor_pair = 0.0
    	
    	for i in range(test_c_patient_no,patient_no):
          	T1 = test_surv_values[i].astype(int)
           	R1 = estimates[i,T1]
            	for j in range(patient_no):
                   	T2 = test_surv_values[j].astype(int)
                    	R2 = estimates[j,T1]
                    	if(T1 < T2):
                        	total_pair += 1.
                        	if(R1 > R2):
                            		cor_pair += 1.
    
    	print("Custom Ctd: {}".format(cor_pair/total_pair))

def test_t(test_uc_surv_values, steps, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out, test_features, test_labels, test_surv_values, fake_labels):		
	test_c_patient_no = test_surv_values.shape[0]-test_uc_surv_values.shape[0]
	test_uc_patient_no = test_uc_surv_values.shape[0]
	test_pairs = []
	for i in range(test_uc_patient_no):
		tmp = np.where(test_uc_surv_values[i] < test_surv_values[:])[0]
		tmp = np.stack([np.ones(tmp.shape[0], dtype=int)*(i+test_c_patient_no), tmp], axis=1)
		test_pairs += [tmp]
	test_pairs = np.vstack(test_pairs).astype(np.int32)
	print(test_c_patient_no)
	print("Test pairs: ", test_pairs.shape[0])
	if(test_pairs.shape[0] == 0): return [0, []]
	return test_algorithm(test_pairs, test_features, test_labels, steps, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out, fake_labels)	

def display_results():
	results = np.genfromtxt('/home/anton/Documents/anton/ICLR/results/SEER/test_full_estimates_x.csv', delimiter=',')

	test_uc_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_uc_surv_time_x.csv', delimiter=',')
	test_c_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER/test_c_surv_time_x.csv', delimiter=',')

	results_uc = results[-test_uc_surv_values.shape[0]:,:]
	results_c = results[:test_c_surv_values.shape[0],:]

	steps = 30
	plt.figure()
	for i in range(0,51,10):
		tmp = results_uc[test_uc_surv_values == i,:]
		
		plt.plot(tmp.mean(axis=0), label='UC_Step {}'.format(i))
		plt.legend()
		plt.xlim([0,54])
		plt.draw()

	plt.figure()
	for i in range(0,51,10):
		tmp = results_c[test_c_surv_values == i,:]
		
		plt.plot(tmp.mean(axis=0), label='C_Step {}'.format(i))
		plt.legend()
		plt.xlim([0,54])
		plt.draw()		
	plt.show()

# display_results()

# hyper parameters
batch_size = 2048	
shuffle = False	
epoch_no = 10
learning_rate = 0.001 #0.0001
lr_decay = 0.1
l1_reg = 0. 
l2_reg = 0.01 #0.001
max_grad = 1	
val_step = 1000
dp_keep = 0.4
# dpt=false, regular dp with keep_prob=0.3
# dpt=true, selu dropout with rate=1-keep_prob=0.6 #0.10/0.05 in the paper
dpt = True #True = 0.6 /// False=0.3
# l_out = np.rint((features_no+steps)/2).astype(int)
l_out = 40
typ = 3
no = 4

train(typ,no, batch_size, shuffle, epoch_no, learning_rate, lr_decay, l1_reg, l2_reg, max_grad, val_step, dp_keep, dpt, l_out)
test(typ ,no, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out)

c_td = np.array([0.717926160832,0.716829512105,0.69854489883,0.715214275085,0.693939667311])	 
print(np.mean(c_td))
print(np.mean(c_td)+np.std(c_td)*1.96)	 
print(np.mean(c_td)-np.std(c_td)*1.96)	
# imports
import numpy as np
import tensorflow as tf
import sys
import selu as se

# algorithm cost function 
def cost_function(pl1,pl2,pl3,pr1,pr2,pr3,labels_left,labels_right,w,w2):
	case = tf.reduce_sum(labels_left, axis=1, keep_dims=True)
	case = tf.tile(case, [1,30])
	w_min = np.amin(w2)

	diff1 = (w_min/w2[0])*tf.multiply(-tf.log(tf.sigmoid(500*(pl1 - pr1))+1e-10)-0.01*tf.log(tf.sigmoid(500*(pl1 - pl2))+1e-10)-0.01*tf.log(tf.sigmoid(500*(pl1 - pl3))+1e-10), labels_left)
	diff2 = (w_min/w2[1])*tf.multiply(-tf.log(tf.sigmoid(500*(pl2 - pr2))+1e-10)-0.01*tf.log(tf.sigmoid(500*(pl2 - pl1))+1e-10)-0.01*tf.log(tf.sigmoid(500*(pl2 - pl3))+1e-10), labels_left/2)
	diff3 = (w_min/w2[2])*tf.multiply(-tf.log(tf.sigmoid(500*(pl3 - pr3))+1e-10)-0.01*tf.log(tf.sigmoid(500*(pl3 - pl1))+1e-10)-0.01*tf.log(tf.sigmoid(500*(pl3 - pl2))+1e-10), labels_left/3)
	diff = tf.where(tf.equal(case,tf.ones_like(diff1)), diff1, diff3)
	diff = tf.where(tf.equal(case,2*tf.ones_like(diff2)), diff2, diff)

	return tf.reduce_sum(tf.multiply(tf.reduce_mean(diff, axis=0), w))
 
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

	h11 = tf.contrib.layers.fully_connected(inputs=data, num_outputs=h, activation_fn=None, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0., scale_l2=0.), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
	h11 = tf.nn.softmax(h11)
	# h10 = tf.divide(h10, tf.tile(tf.reduce_sum(h10, axis=1, keep_dims=True), [1, h]) + 1e-10)
	h11 = tf.cumsum(h11, axis=1)

	h12 = tf.contrib.layers.fully_connected(inputs=data, num_outputs=h, activation_fn=None, weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0., scale_l2=0.), weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
	h12 = tf.nn.softmax(h12)
	# h10 = tf.divide(h10, tf.tile(tf.reduce_sum(h10, axis=1, keep_dims=True), [1, h]) + 1e-10)
	h12 = tf.cumsum(h12, axis=1)

	# layer_out = tf.nn.dropout(layer_in, dp)
	return [h10, h11, h12]	
	
def c_index2(pl1,pl2,pl3,pr1,pr2,pr3,fake_labels):
	case = tf.reduce_sum(fake_labels, axis=1)
	diff1 = tf.reduce_sum(tf.multiply(tf.to_float(pl1 > pr1), fake_labels), axis=1)
	diff2 = tf.reduce_sum(tf.multiply(tf.to_float(pl2 > pr2), fake_labels/2), axis=1)
	diff3 = tf.reduce_sum(tf.multiply(tf.to_float(pl3 > pr3), fake_labels/3), axis=1)

	diff1_1 = tf.where(tf.equal(case,tf.ones_like(diff1)), diff1, tf.zeros_like(diff1))
	diff2_2 = tf.where(tf.equal(case,2*tf.ones_like(diff2)), diff2, tf.zeros_like(diff2))
	diff3_3 = tf.where(tf.equal(case,3*tf.ones_like(diff3)), diff3, tf.zeros_like(diff3))

	return [tf.reduce_sum(diff1_1),tf.reduce_sum(diff2_2),tf.reduce_sum(diff3_3)]
 
def train_algorithm(w2, no, train_pairs, train_features, train_labels, val_pairs, val_features, val_labels, steps, weights, shuffle, batch_size, epoch_no, learning_rate, lr_decay, l1, l2, max_grad, l_out, val_step, dp_rate, dp_type):
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
	[pl1,pl2,pl3] = h_shared(states_left, dp, steps)
	[pr1,pr2,pr3] = h_shared(states_right, dp, steps)

	cost = cost_function(pl1,pl2,pl3, pr1,pr2,pr3,labels_left, labels_right, w, w2)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	trainer = optimizer.minimize(cost)

	c1,c2,c3 = c_index2(pl1,pl2,pl3, pr1,pr2,pr3,labels_left)

	init = tf.global_variables_initializer()
	init2 = tf.local_variables_initializer()
	best_val = -np.inf
	with tf.Session() as sess:
		saver = tf.train.Saver(max_to_keep=1)
		sess.run(init)
		sess.run(init2)
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
				size1 = l_left[np.sum(l_left, axis=1) == 1,:].shape[0]
				size2 = l_left[np.sum(l_left, axis=1) == 2,:].shape[0]
				size3 = l_left[np.sum(l_left, axis=1) == 3,:].shape[0]
				_, a, b1,b2,b3 = sess.run([trainer, cost, c1,c2,c3], feed_dict={lr: learning_rate, dp: dp_rate, features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right, w: weights})
				avg_cost += a / total_batch
				avg_ctd += b1 / total_batch
				sys.stdout.write("--->>> Batch: {}/{}, Loss: {}, Ctd Index 1: {}, Ctd Index 2: {}, Ctd Index 3: {} <<<---\r".format(i, total_batch, a/batch_size, b1/size1, b2/size2, b3/size3))
				sys.stdout.flush()
				
				# validation
				if((i%val_step == 0) & (i > 0)):
					sess.run(init2)
					tot_val1 = 0
					tot_val2 = 0
					tot_val3 = 0
					size1 = 0
					size2 = 0
					size3 = 0
					for k in range(val_parts):
						batch_left = val_features[val_pairs[k*val_batch_size:(k+1)*val_batch_size,0],:]
						batch_right = val_features[val_pairs[k*val_batch_size:(k+1)*val_batch_size,1],:]
						l_left = val_labels[val_pairs[k*val_batch_size:(k+1)*val_batch_size,0],:]
						l_right = val_labels[val_pairs[k*val_batch_size:(k+1)*val_batch_size,1],:]
						size1 += l_left[np.sum(l_left, axis=1) == 1,:].shape[0]
						size2 += l_left[np.sum(l_left, axis=1) == 2,:].shape[0]
						size3 += l_left[np.sum(l_left, axis=1) == 3,:].shape[0]	

						[val_a1,val_a2,val_a3] = sess.run([c1,c2,c3], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
						tot_val1 += val_a1
						tot_val2 += val_a2
						tot_val3 += val_a3
																
					k += 1
					batch_left = val_features[val_pairs[k*val_batch_size:val_no,0],:]
					batch_right = val_features[val_pairs[k*val_batch_size:val_no,1],:]
					l_left = val_labels[val_pairs[k*val_batch_size:val_no,0],:]
					l_right = val_labels[val_pairs[k*val_batch_size:val_no,1],:]
					size1 += l_left[np.sum(l_left, axis=1) == 1,:].shape[0]
					size2 += l_left[np.sum(l_left, axis=1) == 2,:].shape[0]
					size3 += l_left[np.sum(l_left, axis=1) == 3,:].shape[0]		
					[val_a1,val_a2,val_a3] = sess.run([c1,c2,c3], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
					tot_val1 += val_a1
					tot_val2 += val_a2
					tot_val3 += val_a3
										
					print 	
					print("--->>> Inner Val Ctd Index 1: {}, Inner Val Ctd Index 2: {}, Inner Val Ctd Index 3: {}".format(tot_val1/size1,tot_val2/size2,tot_val3/size3))
					if(tot_val1/size1 > best_val):
						print("Saving model...")
						saver.save(sess, logs_path + '/models/model', global_step=epoch)
						best_val = tot_val1/size1
			
			print 
			print("--->>> Epoch: {}/{}".format(epoch+1,epoch_no), ", Loss: ", "{:.9f}".format(avg_cost), "Ctd Index: ", "{:.9f} <<<---".format(avg_ctd))	
  
def test_algorithm(w2, train_pairs, train_features, train_labels, steps, epoch_no, learning_rate, l1, l2, dp_type, l_out, fake_labels):
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
	[pl1,pl2,pl3] = h_shared(states_left, dp, steps)
	[pr1,pr2,pr3] = h_shared(states_right, dp, steps)
	cost = cost_function(pl1,pl2,pl3, pr1,pr2,pr3,labels_left, labels_right, w, w2)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	trainer = optimizer.minimize(cost)

	c1,c2,c3 = c_index2(pl1,pl2,pl3, pr1,pr2,pr3,labels_left)
	with tf.Session() as sess:
		saver = tf.train.Saver(max_to_keep=epoch_no)
		saver.restore(sess,tf.train.latest_checkpoint(logs_path + '/models/'))
		a = 0.		

		[estimates1,estimates2,estimates3] = sess.run([pl1,pl2,pl3], feed_dict={dp: 1., features_left: train_features, features_right: train_features, labels_left: train_labels, labels_right: fake_labels})
		uc_idx = np.unique(train_pairs[:,0])

		k=0
		b=0
		for k in range(train_pairs.shape[0]/bt):
			batch_left = train_features[train_pairs[k*bt:(k+1)*bt,0],:]
			batch_right = train_features[train_pairs[k*bt:(k+1)*bt,1],:]
			l_left = fake_labels[train_pairs[k*bt:(k+1)*bt,0],:]
			l_right = fake_labels[train_pairs[k*bt:(k+1)*bt,1],:]

			[a_t,b_t,c_t] = sess.run([c1,c2,c3], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
			a += a_t
						
			sys.stdout.write("{}/{}\r".format(k,train_pairs.shape[0]/bt))
			sys.stdout.flush()
		k += 1
		batch_left = train_features[train_pairs[k*bt:train_pairs.shape[0],0],:]
		batch_right = train_features[train_pairs[k*bt:train_pairs.shape[0],1],:]
		l_left = fake_labels[train_pairs[k*bt:train_pairs.shape[0],0],:]
		l_right = fake_labels[train_pairs[k*bt:train_pairs.shape[0],1],:]

		[a_t,b_t,c_t] = sess.run([c1,c2,c3], feed_dict={dp: 1., features_left: batch_left, features_right: batch_right, labels_left: l_left, labels_right: l_right})
		a += a_t

	return [a/train_no, estimates1,estimates2,estimates3, b, train_no]

def train(no, batch_size, shuffle, epoch_no, learning_rate, lr_decay, l1_reg, l2_reg, max_grad, val_step, dp_rate, dpt, l_out):
	train_features = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_feature_values_{}.csv'.format(no), delimiter=',', dtype=np.float16)
	train_labels = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_labels_{}.csv'.format(no), delimiter=',').astype(int)
	train_status = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_surv_status_{}.csv'.format(no), delimiter=',').astype(int)
	train_labels[train_status != 0,:] = train_labels[train_status != 0,:]*np.tile(np.expand_dims(train_status[train_status != 0], axis=1), (1,train_labels[train_status != 0,:].shape[1]))
	w2 = np.zeros(3)
	w2[0] = train_status[train_status == 1].shape[0]
	w2[1] = train_status[train_status == 2].shape[0]
	w2[2] = train_status[train_status == 3].shape[0]
	train_uc_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_uc_surv_time_x_{}.csv'.format(no), delimiter=',', dtype=np.float16)
	train_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/train_surv_time_x_{}.csv'.format(no), delimiter=',', dtype=np.float16)
	train_uc_patient_no = train_uc_surv_values.shape[0]
	train_uc_status = train_status[-train_uc_patient_no:]

	train_pairs = []
	train_pairs_cat = np.zeros(30)
	for i in range(train_uc_patient_no):
		# if(train_uc_status[i] == 1):
		if(True):
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

	val_features = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_feature_values_{}.csv'.format(no), delimiter=',', dtype=np.float16)
	val_labels = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_labels_{}.csv'.format(no), delimiter=',').astype(int)
	val_status = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_surv_status_{}.csv'.format(no), delimiter=',').astype(int)
	val_labels[val_status != 0,:] = val_labels[val_status != 0,:]*np.tile(np.expand_dims(val_status[val_status != 0], axis=1), (1,val_labels[val_status != 0,:].shape[1]))

	val_uc_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_uc_surv_time_x_{}.csv'.format(no), delimiter=',', dtype=np.float16)
	val_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/val_surv_time_x_{}.csv'.format(no), delimiter=',', dtype=np.float16)

	val_uc_patient_no = val_uc_surv_values.shape[0]
	val_pairs = []
	for i in range(val_uc_patient_no):
		tmp = np.where(val_uc_surv_values[i] < val_surv_values[:])[0]
		tmp = np.stack([np.ones(tmp.shape[0], dtype=int)*(i+val_surv_values.shape[0]-val_uc_patient_no), tmp], axis=1)
		val_pairs += [tmp.astype(np.int32)]

	val_pairs = np.vstack(val_pairs)

	print("Val pairs: ", val_pairs.shape[0])	

	train_algorithm(w2, no, train_pairs, train_features, train_labels, val_pairs, val_features, val_labels, 30, train_pairs_cat, shuffle, batch_size, epoch_no, learning_rate, lr_decay, l1_reg, l2_reg, max_grad, l_out, val_step, dp_rate, dpt)

def test(no, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out):
	test_uc_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_uc_surv_time_x_{}.csv'.format(no), delimiter=',')

	test_features = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_feature_values_{}.csv'.format(no), delimiter=',')
	test_labels = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_labels_{}.csv'.format(no), delimiter=',').astype(int)
	test_status = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_surv_status_{}.csv'.format(no), delimiter=',').astype(int)
	test_labels[test_status != 0,:] = test_labels[test_status != 0,:]*np.tile(np.expand_dims(test_status[test_status != 0], axis=1), (1,test_labels[test_status != 0,:].shape[1]))
	test_surv_values = np.genfromtxt('/home/anton/Documents/anton/ICLR/data/SEER_dual/test_surv_time_x_{}.csv'.format(no), delimiter=',')

	test_uc_patient_no = test_uc_surv_values.shape[0]
	test_c_patient_no = test_surv_values.shape[0]-test_uc_patient_no

	W = np.zeros(30)
	AUC = np.zeros(30)
	BR = np.zeros(30)
	[Ctd_full, estimates1,estimates2,estimates3, _, _] = test_t(test_uc_surv_values, 30, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out, test_features, test_labels, test_surv_values, test_labels)
	np.savetxt('/home/anton/Documents/anton/ICLR/results/SEER_dual/test_full_estimates1_x_{}.csv'.format(no), estimates1, delimiter=",")
	np.savetxt('/home/anton/Documents/anton/ICLR/results/SEER_dual/test_full_estimates2_x_{}.csv'.format(no), estimates2, delimiter=",")
	np.savetxt('/home/anton/Documents/anton/ICLR/results/SEER_dual/test_full_estimates3_x_{}.csv'.format(no), estimates3, delimiter=",")
	patient_no = test_surv_values.shape[0]

	total_pair = 0.0
    	cor_pair = 0.0
    	
    	for i in range(test_c_patient_no,patient_no):
    		if(test_status[i] == 1):
	          	T1 = test_surv_values[i].astype(int)
	           	R1 = estimates1[i,T1]
	            	for j in range(patient_no):
	                   	T2 = test_surv_values[j].astype(int)
	                    	R2 = estimates1[j,T1]
	                    	if(T1 < T2):
	                        	total_pair += 1.
	                        	if(R1 > R2):
	                            		cor_pair += 1.
    
    	print("Custom Ctd 1: {}".format(cor_pair/total_pair))

    	total_pair = 0.0
    	cor_pair = 0.0
    	
    	for i in range(test_c_patient_no,patient_no):
    		if(test_status[i] == 2):
	          	T1 = test_surv_values[i].astype(int)
	           	R1 = estimates2[i,T1]
	            	for j in range(patient_no):
	                   	T2 = test_surv_values[j].astype(int)
	                    	R2 = estimates2[j,T1]
	                    	if(T1 < T2):
	                        	total_pair += 1.
	                        	if(R1 > R2):
	                            		cor_pair += 1.
    
    	print("Custom Ctd 2: {}".format(cor_pair/total_pair))

    	total_pair = 0.0
    	cor_pair = 0.0
    	
    	for i in range(test_c_patient_no,patient_no):
    		if(test_status[i] == 3):
	          	T1 = test_surv_values[i].astype(int)
	           	R1 = estimates3[i,T1]
	            	for j in range(patient_no):
	                   	T2 = test_surv_values[j].astype(int)
	                    	R2 = estimates3[j,T1]
	                    	if(T1 < T2):
	                        	total_pair += 1.
	                        	if(R1 > R2):
	                            		cor_pair += 1.
    
    	print("Custom Ctd 3: {}".format(cor_pair/total_pair))

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
	return test_algorithm(np.zeros(3), test_pairs, test_features, test_labels, steps, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out, fake_labels)	 

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
l_out = 50
no = 3

train(no, batch_size, shuffle, epoch_no, learning_rate, lr_decay, l1_reg, l2_reg, max_grad, val_step, dp_keep, dpt, l_out)
test(no, epoch_no, learning_rate, l1_reg, l2_reg, dpt, l_out)

c_td = np.array([0.717926160832,0.716829512105,0.69854489883,0.715214275085,0.693939667311])	 
print(np.mean(c_td))
print(np.mean(c_td)+np.std(c_td)*1.96)	 
print(np.mean(c_td)-np.std(c_td)*1.96)	
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from matplotlib import pyplot as plt
learn = tf.contrib.learn
lstm_size = 32  # Lstm中隐藏节点的个数
num_layer = 2  # LSTM的层数
timesteps = 10  # 循环神经网络的截断长度
train_steps = 10000  # 训练轮数
batch_size = 32  # batch大小

training_examples= 10000  # 训练数据个数
testing_examples = 1000  # 测试数据个数
sample_gap = 0.01  # 采样间隔
def generate_data(seq):
	X = []
	Y = []
	for i in range(len(seq) - timesteps - 1):
		X.append([seq[i:i + timesteps]])
		Y.append([seq[i + timesteps]])
	return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def LstmCell():
	lstm_cell = rnn.BasicLSTMCell(lstm_size,state_is_tuple=True)
	return lstm_cell

test_start = training_examples * sample_gap

test_end = (training_examples + testing_examples) * sample_gap


ix = np.sin(np.linspace(0, test_start, training_examples, dtype=np.float32))


train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, training_examples, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, testing_examples, dtype=np.float32)))

x = tf.placeholder(tf.float32,[None,1,timesteps],name='input_x')
print(x.shape)
y_ = tf.placeholder(tf.float32,[None,1],name = 'input_y')
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)
def lstm_cell():
	return tf.contrib.rnn.BasicLSTMCell(lstm_size)

cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layer)])
outputs,final_state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
outputs = outputs[:,-1]
predictions = tf.contrib.layers.fully_connected(outputs,1,activation_fn=tf.tanh)
cost = tf.losses.mean_squared_error(y_,predictions)
optimizer = tf.train.AdamOptimizer().minimize(cost)

def get_batches(X,y,batch_size=64):
	for i in range(0,len(X),batch_size):
		begin_i = i
		end_i = i + batch_size if (i+batch_size) < len(X) else len(X)
		yield X[begin_i:end_i],y[begin_i:end_i]


epochs = 20
session = tf.Session()
with session.as_default() as sess:
	tf.global_variables_initializer().run()
	iteration = 1
	for e in range(epochs):
		print('~'*50)
		for xs,ys in get_batches(train_X,train_y,batch_size):
			print(iteration)
			#feed_dict = {x:xs[:,:,None],y_:ys[:,None],keep_prob:.5}
			feed_dict = {x:xs,y_:ys,keep_prob:1.0}
			loss,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
			if iteration % 100 == 0:
				print('Epochs:{}/{}'.format(e, epochs),
						'Iteration:{}'.format(iteration),
						'Train loss: {:.8f}'.format(loss))
			iteration += 1




with session.as_default() as sess:
	#feed_dict = {x:test_X[:,:,None], keep_prob:1.0}
	feed_dict = {x:test_X, keep_prob:1.0}
	results = sess.run(predictions, feed_dict=feed_dict)
	plt.plot(results,'r', label='predicted')
	plt.plot(test_y, 'g--', label='real sin')
	plt.legend()
	plt.show()

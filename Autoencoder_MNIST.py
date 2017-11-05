import tensorflow as tf 
import numpy as np 

n_input=784
n_hidden_1=256
n_hidden_2=128

X = tf.placeholder('float',[None,784])

init_weight = tf.random_normal_initializer()
init_constant = tf.constant_initializer(0.0)


weights = {
	
	'encoder_w1': tf.get_variable('ew1',shape= [n_input,n_hidden_1],initializer=init_weight),
	'encoder_w2': tf.get_variable('ew2',shape=[n_hidden_1,n_hidden_2],initializer=init_weight),
	'decoder_w1': tf.get_variable('dw1',shape= [n_hidden_2,n_hidden_1],initializer=init_weight),
	'decoder_w2': tf.get_variable('dw2',shape=[n_hidden_1,n_input],initializer=init_weight)
}

bias = {
	
	'encoder_b1' : tf.get_variable('eb1',shape=[n_hidden_1],initializer=init_constant),
	'encoder_b2' : tf.get_variable('eb2',shape=[n_hidden_2],initializer=init_constant),
	'decoder_b1' : tf.get_variable('db1',shape=[n_hidden_1],initializer=init_constant),
	'decoder_b2' : tf.get_variable('db2',shape=[n_input],initializer=init_constant)
}


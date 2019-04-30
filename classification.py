from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True) #从网上下载数据


def add_layer(input,in_size,out_size,activation_function=None):
    #add one more layer and returen the output of this layer
    with tf.name_scope('layer_name'):
        with tf.name_scope('Weight'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram('/weights',Weights)
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram('/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(input,Weights)+biases
        if activation_function is None: #默认有显示
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        tf.summary.histogram('/outputs', outputs)

        return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #是一个百分比
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784]) #28*28
ys=tf.placeholder(tf.float32,[None ,10])

#add output layer
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)# softmax 一般用于分类问题

#the error between prediction and  real data
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) #loss

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()

#important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))



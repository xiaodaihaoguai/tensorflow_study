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
    y_pre=sess.run(prediction,feed_dict={xs:v_xs, keep_prob: 1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #是一个百分比
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)#从截断的正态分布中输出随机值。 shape表示生成张量的维度，mean是均值，stddev是标准差
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return  tf.Variable(initial)

def conv2d(x,W):#x是值 图片的所有信息
    #stride[1,x-movement,y-movement,1],首尾都是1
    # valid padding 按照在图片内padding抽取的要小一些，same padding图外不0抽取和原图相同
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):#pooling 防止跨度太大损失信息
    #ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    #第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是featuremap，依然是[batch, height, width, channels]这样的shape
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #stride 是缩减的补偿


#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784]) #28*28
ys=tf.placeholder(tf.float32,[None ,10])
keep_prob=tf.placeholder(tf.float32)#每个元素被保留的概率，那么 keep_prob:1就是所有元素全部保留的意思。
一般在大量数据训练时，为了防止过拟合，添加Dropout层，设置一个0~1之间的小数
x_image=tf.reshape(xs,[-1,28,28,1]) #【n_sampels,28,28,1】黑白 通道数 是1


##conv1 layer##
#首先在每个5x5网格中，提取出32张特征图。其中weight_variable中前两维是指网格的大小，第三维的1是指输入通道数目，
# 第四维的32是指输出通道数目（也可以理解为使用的卷积核个数、得到的特征图张数）。每个输出通道都有一个偏置项，因此偏置项个数为32
W_conv1=weight_variable([5,5,1,32]) #patch（kernel） 5x5 insize=1输入的通道数（图像的厚度）,outsize=32 输出通道数
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output size 28x28x32
h_pool1=max_pool_2x2(h_conv1) #output size 14x14x32

##conv2 layer##
W_conv2=weight_variable([5,5,32,64]) #patch（kernel） 5x5 insize=32输入的通道数（图像的gao度）,outsize=64 输出通道数
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output size 14x14x64
h_pool2=max_pool_2x2(h_conv2) #output size 7x7x64

##func1 layer##
W_fc1=weight_variable([7*7*64,1024])  #1024??? suibian
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#-1 all samples  [n_samples,7,7,64]>>[n_samples,7*7*64]
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_dropout=tf.nn.dropout(h_fc1,keep_prob)

##func2 layer##
W_fc2=weight_variable([1024,10])   # 10分类
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_dropout,W_fc2)+b_fc2)


#add output layer
#prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)# softmax 一般用于分类问题

#the error between prediction and  real data
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) #loss

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.Session()

#important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))


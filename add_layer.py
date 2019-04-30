import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input,in_size,out_size,n_layer,activation_function=None):
    #add one more layer and returen the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weight'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(input,Weights)+biases
        if activation_function is None: #默认有显示
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)

        return outputs

#make up some real data
x_data=np.linspace(-1,1,300)[:,np.newaxis] #加一个维度
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

#define placeholder for inputs to network
with tf.name_scope('inputs'):#囊括在一个图层窗口当中
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')#None 表示传入多少个数都ok

#要构建一个输入层有1个神经元，隐藏层有10个神经元，输出层有1个神经元的神经网络
#add hidden layer
l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
#add output layer
prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)

#the error between prediction and real data
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1])) #有一个reduction_indices参数，表示函数的处理维度
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss) #使用优化器，以学习率为0.1的速度，对loss进行最小化的优化

#important step
init=tf.initialize_all_variables()# 很重要 初始化变量


sess=tf.Session()
merged=tf.summary.merge_all()#回收所有summary
writer=tf.summary.FileWriter("logs/", sess.graph) #定义好session后再定义，把整个框架loading到一个文件夹里，然后再用浏览器观看
# 在log的上一级目录运行tensorboard --logdir='logs/' 打开网址

sess.run(init)




# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)#编号
ax.scatter(x_data, y_data)#把数据plot出来
plt.ion()#本次运行请注释，全局运行不要注释, 让plt.show 每一次会让程序暂停 显示完向下走
plt.show()




for i in range(1000):
    #trainning
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})#feed_dict在传入数据,这么做的原因 在梯度下降的时候可能用小部分的数据更有效率
    #to see the improvment
    if i%50==0:
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)#将loss的每个值写进文件
        # print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))
        # prediction_value=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        # lines=ax.plot(x_data,prediction_value,'r-',lw=5)#
        try:
            ax.lines.remove(lines[0]) #画下一条线之前要删除line的第一个单位，try是防止之前没有线报错
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)#红色线 宽度为5
        plt.pause(0.1)


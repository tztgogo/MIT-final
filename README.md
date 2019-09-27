# MIT-final


## 介绍

本次项目我们主要是为了完成机器的手写体识别，运用的是mnist算法

## Mnist

mnist算法实际上是Lenet5神经网络算法，其中包含两个卷积层，两个池化层，和三个全连接层，其原理便是输入一张28*28像素的图片，然后经过卷积和池化后，通过全连接层
后输出一个十维向量，对应着0-9可能性的概率。

首先我们构建了mnist模型的代码，然后用mnist提供的训练数据集来训练模型权值，准确率可以高达99%
然后我们用 saver.save()将模型存储了下来，得到四个文件，分别为一个checkpoint文件和三个model.ckpt文件。
最后我们开始读入模型，运用 TensorFlow的restore（）函数，给一个图片然后预测它是几

## 代码实现过程：

### 运行mnist模型：
首先是mnist的模型部分：

'''python
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
'''

然后是mnist的测试部分：
首先打开一个tensorflow操作面：

'''python
with tf.Session() as sess
'''

然后给定操作步长：

'''python
batch = mnist.train.next_batch(50)
'''

然后开始正式训练：

'''python
train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
'''
最后返回一个模型准确率：

'''python
print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
'''

### 保存模型部分：
保存模型的代码：

'''python
saver = tf.train.Saver()
'''

其保存的路径和文件的名称：

'''python
saver.save(sess, './model.ckpt')
'''

最后你可以得到这么四个文件：
![image]()

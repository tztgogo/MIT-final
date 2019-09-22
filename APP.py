{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image, ImageFilter\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt    \n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def image_prepare(image_path):\n",
    "    img = Image.open(image_path)\n",
    "    im = img.resize((28, 28), Image.ANTIALIAS)\n",
    "    im = im.convert('L')\n",
    "    tv = list(im.getdata()) \n",
    "    tva = [(255-x)*1.0/255.0 for x in tv] \n",
    "    \n",
    "    \n",
    "    return tva"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Please enter the image path :  ./WechatIMG2.png\n"
     ]
    }
   ],
   "source": [
    "image_path = input('Please enter the image path : ')\n",
    "\n",
    "\n",
    "result = image_prepare(image_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = tf.placeholder(tf.float32, [None, 784])\n",
    "\n",
    "y_ = tf.placeholder(tf.float32, [None, 10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def weight_variable(shape):\n",
    "    initial = tf.truncated_normal(shape, stddev = 0.1)\n",
    "    return tf.Variable(initial)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def bias_variable(shape):\n",
    "    initial = tf.constant(0.1, shape = shape)\n",
    "    return tf.Variable(initial)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def conv2d(x, W):\n",
    "    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = \"SAME\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def max_pool_2x2(x):\n",
    "    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = \"SAME\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Logging before flag parsing goes to stderr.\n",
      "W0922 18:53:41.462507 4380644800 deprecation.py:506] From <ipython-input-14-ebaa80ae0bb6>:22: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n"
     ]
    }
   ],
   "source": [
    "W_conv1 = weight_variable([5,5,1,32])\n",
    "b_conv1 = bias_variable([32])\n",
    "\n",
    "x_image = tf.reshape(x, [-1,28,28,1])\n",
    "\n",
    "h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)\n",
    "h_pool1 = max_pool_2x2(h_conv1)\n",
    "\n",
    "W_conv2 = weight_variable([5,5,32,64])\n",
    "b_conv2 = bias_variable([64])\n",
    "\n",
    "h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2)+b_conv2)\n",
    "h_pool2 = max_pool_2x2(h_conv2)\n",
    "\n",
    "W_fc1 = weight_variable([7*7*64, 1024])\n",
    "b_fc1 = bias_variable([1024])\n",
    "\n",
    "h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])\n",
    "h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)\n",
    "\n",
    "keep_prob = tf.placeholder(\"float\")\n",
    "h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)\n",
    "\n",
    "W_fc2 = weight_variable([1024, 10])\n",
    "b_fc2 = bias_variable([10])\n",
    "\n",
    "y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))\n",
    "train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "saver = tf.train.Saver()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0922 18:53:43.994906 4380644800 deprecation.py:323] From /Users/tanzhentao/anaconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py:1276: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use standard file APIs to check for files with this prefix.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "识别结果:\n",
      "1\n"
     ]
    }
   ],
   "source": [
    "with tf.Session() as sess:\n",
    "    sess.run(tf.global_variables_initializer())\n",
    "    saver.restore(sess, \"./model.ckpt\")\n",
    "    \n",
    "    prediction = tf.argmax(y_conv,1)\n",
    "    predict = prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session = sess)\n",
    "\n",
    "    print('识别结果:')\n",
    "    print(predict[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

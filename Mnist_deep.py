{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import absolute_import\n",
    "from __future__ import division\n",
    "from __future__ import print_function\n",
    "\n",
    "import argparse\n",
    "import sys\n",
    "import tempfile\n",
    "\n",
    "from tensorflow.python import pywrap_tensorflow\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "\n",
    "import tensorflow as tf\n",
    "\n",
    "FLAGS = None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def deepnn(x):\n",
    "    \n",
    "  \"\"\"deepnn builds the graph for a deep net for classifying digits.\n",
    "  Args:\n",
    "    x: an input tensor with the dimensions (N_examples, 784), where 784 is the\n",
    "    number of pixels in a standard MNIST image.\n",
    "  Returns:\n",
    "    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values\n",
    "    equal to the logits of classifying the digit into one of 10 classes (the\n",
    "    digits 0-9). keep_prob is a scalar placeholder for the probability of\n",
    "    dropout.\n",
    "  \"\"\"\n",
    "  # Reshape to use within a convolutional neural net.\n",
    "  # Last dimension is for \"features\" - there is only one here, since images are\n",
    "  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.\n",
    "  x_image = tf.reshape(x, [-1, 28, 28, 1])\n",
    "\n",
    "  # First convolutional layer - maps one grayscale image to 32 feature maps.\n",
    "  W_conv1 = weight_variable([5, 5, 1, 32])\n",
    "  b_conv1 = bias_variable([32])\n",
    "  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)\n",
    "\n",
    "  # Pooling layer - downsamples by 2X.\n",
    "  \n",
    "  h_pool1 = max_pool_2x2(h_conv1)\n",
    "\n",
    "  # Second convolutional layer -- maps 32 feature maps to 64.\n",
    "  W_conv2 = weight_variable([5, 5, 32, 64])\n",
    "  b_conv2 = bias_variable([64])\n",
    "  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)\n",
    "\n",
    "  # Second pooling layer.\n",
    " \n",
    "  h_pool2 = max_pool_2x2(h_conv2)\n",
    "\n",
    "  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image\n",
    "  # is down to 7x7x64 feature maps -- maps this to 1024 features.\n",
    " \n",
    "  W_fc1 = weight_variable([7 * 7 * 64, 1024])\n",
    "  b_fc1 = bias_variable([1024])\n",
    "\n",
    "  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])\n",
    "  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)\n",
    "\n",
    "  # Dropout - controls the complexity of the model, prevents co-adaptation of\n",
    "  # features.\n",
    "  \n",
    "  keep_prob = tf.placeholder(tf.float32)\n",
    "  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)\n",
    "\n",
    "  # Map the 1024 features to 10 classes, one for each digit\n",
    "  \n",
    "  W_fc2 = weight_variable([1024, 10])\n",
    "  b_fc2 = bias_variable([10])\n",
    "\n",
    "  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2\n",
    "    \n",
    "  return y_conv, keep_prob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def conv2d(x, W):\n",
    "  \"\"\"conv2d returns a 2d convolution layer with full stride.\"\"\"\n",
    "  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def max_pool_2x2(x):\n",
    "  \"\"\"max_pool_2x2 downsamples a feature map by 2X.\"\"\"\n",
    "  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],\n",
    "                        strides=[1, 2, 2, 1], padding='SAME')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def weight_variable(shape):\n",
    "  \"\"\"weight_variable generates a weight variable of a given shape.\"\"\"\n",
    "  initial = tf.truncated_normal(shape, stddev=0.1)\n",
    "  return tf.Variable(initial)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def bias_variable(shape):\n",
    "  \"\"\"bias_variable generates a bias variable of a given shape.\"\"\"\n",
    "  initial = tf.constant(0.1, shape=shape)\n",
    "  return tf.Variable(initial)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main(_):\n",
    "  # Import data\n",
    "  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)\n",
    "\n",
    "  # Create the model\n",
    "  x = tf.placeholder(tf.float32, [None, 784])\n",
    "\n",
    "  # Define loss and optimizer\n",
    "  y_ = tf.placeholder(tf.float32, [None, 10])\n",
    "\n",
    "  # Build the graph for the deep net\n",
    "  y_conv, keep_prob = deepnn(x)\n",
    "\n",
    " \n",
    "  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,\n",
    "                                                            logits=y_conv)\n",
    "  cross_entropy = tf.reduce_mean(cross_entropy)\n",
    "\n",
    "  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)\n",
    "\n",
    "\n",
    "  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))\n",
    "  correct_prediction = tf.cast(correct_prediction, tf.float32)\n",
    "  accuracy = tf.reduce_mean(correct_prediction)\n",
    "\n",
    "  graph_location = tempfile.mkdtemp()\n",
    "  print('Saving graph to: %s' % graph_location)\n",
    "  train_writer = tf.summary.FileWriter(graph_location)\n",
    "  train_writer.add_graph(tf.get_default_graph())\n",
    "\n",
    "  saver = tf.train.Saver()\n",
    "    \n",
    "  with tf.Session() as sess:\n",
    "    sess.run(tf.global_variables_initializer())\n",
    "    for i in range(600):\n",
    "      batch = mnist.train.next_batch(50)\n",
    "      if i % 100 == 0:\n",
    "        train_accuracy = accuracy.eval(feed_dict={\n",
    "            x: batch[0], y_: batch[1], keep_prob: 1.0})\n",
    "        print('step %d, training accuracy %g' % (i, train_accuracy))\n",
    "      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})\n",
    "    saver.save(sess, './model.ckpt')\n",
    "    print('test accuracy %g' % accuracy.eval(feed_dict={\n",
    "        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0922 18:49:25.580255 4414387648 deprecation.py:323] From <ipython-input-7-9134c91feb6a>:3: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.\n",
      "W0922 18:49:25.581241 4414387648 deprecation.py:323] From /Users/tanzhentao/anaconda3/lib/python3.7/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please write your own downloading logic.\n",
      "W0922 18:49:25.581900 4414387648 deprecation.py:323] From /Users/tanzhentao/anaconda3/lib/python3.7/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0922 18:49:25.791764 4414387648 deprecation.py:323] From /Users/tanzhentao/anaconda3/lib/python3.7/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "W0922 18:49:25.793108 4414387648 deprecation.py:323] From /Users/tanzhentao/anaconda3/lib/python3.7/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.one_hot on tensors.\n",
      "W0922 18:49:25.833285 4414387648 deprecation.py:323] From /Users/tanzhentao/anaconda3/lib/python3.7/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz\n",
      "Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz\n",
      "Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0922 18:49:26.065526 4414387648 deprecation.py:506] From <ipython-input-2-34c8153d9c20>:49: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n",
      "W0922 18:49:26.082863 4414387648 deprecation.py:323] From <ipython-input-7-9134c91feb6a>:16: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "\n",
      "Future major versions of TensorFlow will allow gradients to flow\n",
      "into the labels input on backprop by default.\n",
      "\n",
      "See `tf.nn.softmax_cross_entropy_with_logits_v2`.\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving graph to: /var/folders/yl/5396lngd0dggmzkpzw40jm4r0000gn/T/tmptym3g6x_\n",
      "step 0, training accuracy 0.14\n",
      "step 100, training accuracy 0.88\n",
      "step 200, training accuracy 0.94\n",
      "step 300, training accuracy 0.92\n",
      "step 400, training accuracy 0.92\n",
      "step 500, training accuracy 0.96\n",
      "test accuracy 0.9466\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[0;31mSystemExit\u001b[0m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/tanzhentao/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3333: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "  parser = argparse.ArgumentParser()\n",
    "  parser.add_argument('--data_dir', type=str,\n",
    "                      default='/tmp/tensorflow/mnist/input_data',\n",
    "                      help='Directory for storing input data')\n",
    "  FLAGS, unparsed = parser.parse_known_args()\n",
    "  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)"
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

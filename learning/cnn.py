import tensorflow as tf
import nibabel as nib
import sys
from os import listdir
from os.path import isfile, join


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Performs 2D convolution of input x, with filter W.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Performs 2x2 max pooling to reduce dimensionality
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 32,32])
y_ = tf.placeholder(tf.float32, [None, 2])

# Reshape x as needed for conv2d function: [batch #, width, height, # color channels]
x_image = tf.reshape(x, [-1, 32, 32, 1])

# We want to produce 32 features for each pixel's 5x5-pixel patch by convolving filter W_conv1 with input x.
# W_conv1 shape: [filter height, filter width, # in channels, # out channels]
W_conv1 = weight_variable([5, 5, 1, 32])
# Each neuron will have a bias attached to each out-channel of the convolution
b_conv1 = bias_variable([32])
# Apply ReLU (aka. max(0, x)) to the result of convolve(x,W)+b
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Reduce dimensionality of the images to 14x14 by performing 2x2 max pooling.
# Helps trim number of total features and reduce overfitting.
h_pool1 = max_pool_2x2(h_conv1)

# Another set of convolution+ReLU+pool layers.
# Input is the 32 output features for each of the 16x16 pixels from first layer.
# Output is 64 new features for each pixel of the now-8x8 image (resized with pooling).
# Performs same 5x5 convolution.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


# TODO: Right now, loading a "random" 32x32 chunk for each image as its training data.
# Eventually, this should be using chunks containing lesions (probably... ?)


training_images = [join(sys.argv[1], f) for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
training_data = []
training_labels = []
for f in training_images:
    print 'Loading training image %s' % f
    mri = nib.load(f)
    mri_data = mri.get_data()
    mri_shape = mri_data.shape
    w = mri_shape[0]
    h = mri_shape[1]
    slices = mri_shape[2]
    training_data.append(mri_data[w/2:w/2+32, h/2:h/2+32, slices/2])
    # TODO: actually append the correct (one-hot formatted) label for this image
    training_labels.append([0, 1])

print 'Running training...'
train_step.run(feed_dict={x: training_data, y_: training_labels, keep_prob: 0.5})


test_images = [join(sys.argv[2], f) for f in listdir(sys.argv[2]) if isfile(join(sys.argv[2], f))]
test_data = []
test_labels = []
for f in test_images:
    print 'Loading test image %s' % f
    mri = nib.load(f)
    mri_data = mri.get_data()
    mri_shape = mri_data.shape
    w = mri_shape[0]
    h = mri_shape[1]
    slices = mri_shape[2]
    test_data.append(mri_data[w/2:w/2+32, h/2:h/2+32, slices/2])
    # TODO: actually append the correct (one-hot formatted) label for this image
    test_labels.append([1, 0])

print("Accuracy %g" % accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))

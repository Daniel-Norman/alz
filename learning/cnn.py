import tensorflow as tf
import nibabel as nib
import sys
import csv
import random
import numpy as np
from os import listdir
from os.path import isfile, join
from skimage.transform import resize

if len(sys.argv) != 4:
    print 'Expects 3 arguments: n_iterations label_csv data_directory'
    quit()

# Google TensorFlow-provided CNN code:

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


# Our code:

# For each WML region in the image, add those pixels (resized to 32x32) as a new data point with
# the label associated with the image's patient's cognitive impairment
def extract_roi_from_csv(lesions_csv, image, label):
    data = []
    labels = []

    with open(lesions_csv, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            roi = []
            for number in row:
                roi.append(int(number))
            region = image[roi[1]:roi[3], roi[2]:roi[4], roi[0]]
            data.append(resize(region, (32, 32)))
            labels.append(label)
    return data, labels


# Load the label associated with each image
def setup_labels(label_file):
    labels_per_hist = {}
    with open(label_file) as lbl_csv:
        reader = csv.reader(lbl_csv)
        for row in reader:
            labels_per_hist[row[0]] = int(row[1])
    return labels_per_hist


def read_data(labels_per_file, directory):
    training_data = []
    training_labels = []
    test_data = []
    test_labels = []

    data = []
    labels = []

    images = [f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.nii')]

    for f in images:
        sample_number = f.replace('.nii', '')
        if sample_number not in labels_per_file:
            continue

        image_file = join(directory, f)
        mri = nib.load(image_file)
        mri_data = mri.get_data()
        lesions_file = join(directory, 'lesions_%s.csv' % f.replace('.nii', ''))
        label = labels_per_file[sample_number]
        d, l = extract_roi_from_csv(lesions_file, mri_data, [0, 1] if label == 0 else [1, 0])

        if random.random() < 0.2:
            test_data.extend(d)
            test_labels.extend(l)
        else:
            training_data.extend(d)
            training_labels.extend(l)

        data.extend(d)
        labels.extend(l)

    return data, labels

    #return training_data, training_labels, test_data, test_labels

print 'Setting up labels...'
label_map = setup_labels(sys.argv[2])

print 'Loading data...'
data, labels = read_data(labels_per_file=label_map, directory=sys.argv[3])
print 'Loaded %s lesion images.' % len(labels)
accuracies = []

for i in xrange(int(sys.argv[1])):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    training_data = []
    training_labels = []
    test_data = []
    test_labels = []

    for j in xrange(len(labels)):
        if random.random() < 0.2:
            test_data.append(data[j])
            test_labels.append(labels[j])
        else:
            training_data.append(data[j])
            training_labels.append(labels[j])

    # print 'Loaded %s test samples and %s test samples.' % (len(training_labels), len(test_labels))

    print 'Running training iteration %s...' % i
    train_step.run(feed_dict={x: training_data, y_: training_labels, keep_prob: 0.5})

    accuracies.append(accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))
    print 'Average accuracy: %.4f (+/- %.4f)' % (np.array(accuracies).mean(), 2 * np.array(accuracies).std())
    # print("Accuracy %g" % )

accuracies = np.array(accuracies)
print 'Average accuracy: %.4f (+/- %.4f)' % (accuracies.mean(), 2*accuracies.std())

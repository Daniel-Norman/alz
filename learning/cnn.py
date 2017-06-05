import tensorflow as tf
import nibabel as nib
import sys
import csv
from os import listdir
from os.path import isfile, join
from skimage.transform import resize


training_directory = sys.argv[1]
training_labels_csv = sys.argv[2]
test_directory = sys.argv[3]
test_labels_csv = sys.argv[4]


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

def extract_roi_csv(input_file):
    regions_of_interest = []
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_list = []
            for number in row:
                row_list.append(int(number))
            regions_of_interest.append(row_list)
    return regions_of_interest


# For each WML region in the image, add those pixels (resized to 32x32) as a new data point with
# the label associated with the image's patient's cognitive impairment
def add_regions_to_data(data, labels, image, regions_of_interest, label):
    for roi in regions_of_interest:
        region = image[roi[1]:roi[3], roi[2]:roi[4], roi[0]]
        data.append(resize(region, (32, 32)))
        labels.append(label)


def read_data(data, labels, label_file, directory):
    labels_per_image = {}
    images = [f for f in listdir(directory) if isfile(join(directory, f)) and not f.endswith("csv")]

    # Load the label associated with each image
    with open(label_file) as label_csv:
        reader = csv.reader(label_csv)
        for row in reader:
            labels_per_image[row[0]] = int(row[1])

    # For each image, load the CSV of lesion regions associated with it and add these regions to the data.
    # Every data point for a particular image shares the same label
    for f in images:
        image_file = join(directory, f)
        csv_file = join(directory, 'lesions_masked_%s.csv' % f)
        mri = nib.load(image_file)
        mri_data = mri.get_data()
        regions_of_interest = extract_roi_csv(csv_file)
        label = [1, 0] if labels_per_image[f] == 0 else [0, 1]
        print 'Loading image %s, CSV %s, label %s' % (image_file, csv_file, label)
        add_regions_to_data(data, labels, mri_data, regions_of_interest, label)


training_data = []
training_labels = []
print 'Loading training data...'
read_data(training_data, training_labels, training_labels_csv, training_directory)


print 'Running training...'
train_step.run(feed_dict={x: training_data, y_: training_labels, keep_prob: 0.5})


test_data = []
test_labels = []
print 'Loading test data...'
read_data(test_data, test_labels, test_labels_csv, test_directory)

print("Accuracy %g" % accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))

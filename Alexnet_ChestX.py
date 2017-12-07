import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('Training set', mnist.train.images.shape, mnist.train.labels.shape)
print('Test set', mnist.test.images.shape, mnist.test.labels.shape)

# Training Parameters
epoch_num = 2
learning_rate = 0.001
# training_iters = 250
batch_size = 128
display_step = 50
training_iters = mnist.train.images.shape[0] // batch_size

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout_keep = 0.75  # Dropout, probability to keep units

# Graph Input
x = tf.placeholder("float32", [None, n_input], name="x_input")
y = tf.placeholder("float32", [None, n_classes], name="y_input")
keep_prob = tf.placeholder("float32")

# Conv2D
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# MaxPool2D
def maxpool2d(x, strides=2, ksize=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding="VALID")

# Normalization
def lr_normalization(x, alpha=1e-4, beta=0.75):
    return tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=alpha, beta=beta)

# Fully_connected
def f_connected(x, w, b, dropout_keep):
    if len(x.get_shape().as_list()) == 4:
        # Reshape conv2 output to fit fully connected layer input
        x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = tf.add(tf.matmul(x, w), b)
    x = tf.nn.relu(x)
    return tf.nn.dropout(x, dropout_keep)

# Create model
def alex_net(x, weights, biases, dropout_keep):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    with tf.name_scope("conv1"):
        conv1 = conv2d(x, weights["wc1"], biases["bc1"])
        # Normalization
        conv1 = lr_normalization(conv1)
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, strides=1, ksize=2)

    # Convolution Layer
    with tf.name_scope("conv2"):
        conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
        # Normalization
        conv2 = lr_normalization(conv2)
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, strides=2, ksize=3)

    # Convolution Layer
    with tf.name_scope("conv3"):
        conv3 = conv2d(conv2, weights["wc3"], biases["bc3"])

    # Convolution Layer
    with tf.name_scope("conv4"):
        conv4 = conv2d(conv3, weights["wc4"], biases["bc4"])

    # Convolution Layer
    with tf.name_scope("conv5"):
        conv5 = conv2d(conv4, weights["wc5"], biases["bc5"])
        # Max Pooling (down-sampling)
        conv5 = maxpool2d(conv5, strides=2, ksize=3)

    # Fully connected layer
    with tf.name_scope("fcon1"):
        fc1 = f_connected(conv5, weights["wf1"], biases["bf1"], dropout_keep)

    # Fully connected layer
    with tf.name_scope("fcon2"):
        fc2 = f_connected(fc1, weights["wf2"], biases["bf2"], dropout_keep)

    # Output, class prediction
    with tf.name_scope("fcon3"):
        out = tf.add(tf.matmul(fc2, weights["out"]), biases["out"])
        return out

# Store layers weight & bias
weights = {
    # 7x7 conv, 1 input, 64 outputs
    "wc1": tf.Variable(tf.random_normal([7, 7, 1, 64])),
    # 5x5 conv, 64 inputs, 128 outputs
    "wc2": tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # 3x3 conv, 128 inputs, 256 outputs
    "wc3": tf.Variable(tf.random_normal([3, 3, 128, 256])),
    # 3x3 conv, 256 inputs, 256 outputs
    "wc4": tf.Variable(tf.random_normal([3, 3, 256, 256])),
    # 3x3 conv, 256 inputs, 128 outputs
    "wc5": tf.Variable(tf.random_normal([3, 3, 256, 128])),
    # fully connected, 6*6*128 inputs, 1024 outputs
    "wf1": tf.Variable(tf.random_normal([6 * 6 * 128, 1024])),
    # fully connected, 1024 inputs, 1024 outputs
    "wf2": tf.Variable(tf.random_normal([1024, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    "out": tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    "bc1": tf.Variable(tf.random_normal([64])),
    "bc2": tf.Variable(tf.random_normal([128])),
    "bc3": tf.Variable(tf.random_normal([256])),
    "bc4": tf.Variable(tf.random_normal([256])),
    "bc5": tf.Variable(tf.random_normal([128])),
    "bf1": tf.Variable(tf.random_normal([1024])),
    "bf2": tf.Variable(tf.random_normal([1024])),
    "out": tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = alex_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope("Training"):
    with tf.name_scope("total_loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar("total_loss", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope("Evaluate_model"):
    with tf.name_scope("correct_pred"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

# predict the class of the sample
pred_class = tf.argmax(pred, axis=1)

# Initializing the variables
init = tf.global_variables_initializer()

# make a figure to show the loss
def makeFig(ax, xlist, ylist):
    ax.scatter(xlist, ylist)


# creating the figure and adjusting its settings
fig, axs = plt.subplots(nrows=1, ncols=2, sharex="col")
plt.ion()  # enable interactivity
fig.set_size_inches(10, 5)
ax0 = axs[0]
ax0.set_ylabel("Training Loss")
ax0.set_xlabel("step")
ax1 = axs[1]
ax1.set_ylim([0, 1])
ax1.set_ylabel("Training Accuracy")
ax1.set_xlabel("step")

xList = list()
lossList = list()
accList = list()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Write summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("./Output" + "/train/", sess.graph)
    valid_writer = tf.summary.FileWriter("./Output" + "/valid/")
    for epoch in range(epoch_num):
        for step in range(training_iters):
            total_step = epoch * training_iters + step #total step
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            summary, _ =  sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout_keep})
            train_writer.add_summary(summary, total_step)

            if total_step % display_step == 0:
                # Calculate batch loss and accuracy
                summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                valid_writer.add_summary(summary, total_step)
                print("Iter " + str(epoch * training_iters + step) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))
                xList.append(total_step)
                lossList.append(loss)
                accList.append(acc)
                # Plot loss
                makeFig(ax0, xList, lossList)
                # plt.draw()
                plt.pause(0.001)
                # Plot accuracy
                makeFig(ax1, xList, accList)
                plt.draw()

    # Saving the figure at the end
    fig.savefig("training_curves.png")
    print("Optimization Finished!")

    # test the network
    # Calculate accuracy
    acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                        y: mnist.test.labels[:256],
                                        keep_prob: 1.})
    print("Testing Accuracy:", acc_test)
    # Predict single images
    n_images = 4
    test_images = mnist.test.images[:n_images]
    preds = sess.run(pred_class, feed_dict={x: test_images,
                                            keep_prob: 1.})
    # Display
    for i in range(n_images):
        plt.figure()
        plt.imshow(np.reshape(test_images[i], [28, 28]), cmap="gray")
        plt.show()
        print("Model prediction:", preds[i])

    train_writer.close()
    valid_writer.close()
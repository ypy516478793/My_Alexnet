import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py

h5f_train = h5py.File('/data/hula/pyuan2/ChestX_data/chest256_train_801010.h5', 'r')
X_train = h5f_train['X_train'][:]
Y_train = h5f_train['Y_train'][:]
h5f_train.close()

h5f_valid = h5py.File('/data/hula/pyuan2/ChestX_data/chest256_val_801010.h5', 'r')
X_valid = h5f_valid['X_val'][:]
Y_valid = h5f_valid['Y_val'][:]
h5f_valid.close()

h5f_test = h5py.File('/data/hula/pyuan2/ChestX_data/chest256_test_801010.h5', 'r')
X_test = h5f_test['X_test'][:]
Y_test = h5f_test['Y_test'][:]
h5f_test.close()


print('Training set', X_train.shape, Y_train.shape)
print('Test set', X_test.shape, Y_test.shape)

# Training Parameters
epoch_num = 100
learning_rate = 0.001
# training_iters = 250
batch_size = 50
display_step = 50
training_iters = X_train.shape[0] // batch_size

# Network Parameters
n_input1 = 256  # data input (img shape: 256*256)
n_input2 = 256
n_classes = 15  # labels total classes
dropout_keep = 0.75  # Dropout, probability to keep units

# Graph Input
x = tf.placeholder("float32", [None, n_input1, n_input2], name="x_input")
y = tf.placeholder("float32", [None, n_classes], name="y_input")
keep_prob = tf.placeholder("float32")

# Conv2D
def conv2d(x, W, b, strides=1, padding="SAME"):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# MaxPool2D
def maxpool2d(x, strides=2, ksize=2, padding="VALID"):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding)

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
    x = tf.reshape(x, shape=[-1, 256, 256, 1])

    # Convolution Layer
    with tf.name_scope("conv1"):
        conv1 = conv2d(x, weights["wc1"], biases["bc1"], strides=4,  padding="VALID")
        # Normalization
        conv1 = lr_normalization(conv1)
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, strides=2, ksize=3, padding="SAME")

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
        conv5 = conv2d(conv4, weights["wc5"], biases["bc5"], padding="VALID")
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
    "wc1": tf.Variable(tf.random_normal([12, 12, 1, 96])),
    "wc2": tf.Variable(tf.random_normal([5, 5, 96, 256])),
    "wc3": tf.Variable(tf.random_normal([3, 3, 256, 384])),
    "wc4": tf.Variable(tf.random_normal([3, 3, 384, 384])),
    "wc5": tf.Variable(tf.random_normal([3, 3, 384, 256])),
    "wf1": tf.Variable(tf.random_normal([6 * 6 * 256, 4096])),
    "wf2": tf.Variable(tf.random_normal([4096, 4096])),
    "out": tf.Variable(tf.random_normal([4096, n_classes]))
}

biases = {
    "bc1": tf.Variable(tf.random_normal([96])),
    "bc2": tf.Variable(tf.random_normal([256])),
    "bc3": tf.Variable(tf.random_normal([384])),
    "bc4": tf.Variable(tf.random_normal([384])),
    "bc5": tf.Variable(tf.random_normal([256])),
    "bf1": tf.Variable(tf.random_normal([4096])),
    "bf2": tf.Variable(tf.random_normal([4096])),
    "out": tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = alex_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope("Training"):
    with tf.name_scope("total_loss"):
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar("total_loss", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope("Evaluate_model"):
    with tf.name_scope("correct_pred"):
        pred_bool = tf.equal(tf.round(tf.nn.sigmoid(pred)), y)
        correct_pred = tf.reduce_prod(tf.cast(pred_bool,tf.float32), 1)
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(correct_pred)
        tf.summary.scalar("accuracy", accuracy)

# predict the labels of the sample
pred_labels = tf.round(tf.nn.sigmoid(pred))

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
    train_writer = tf.summary.FileWriter("./Output_chestX" + "/train/", sess.graph)
    valid_writer = tf.summary.FileWriter("./Output_chestX" + "/valid/")
    for epoch in range(epoch_num):
        for step in range(training_iters):
            total_step = epoch * training_iters + step #total step
            batch_x = X_train[step * batch_size: step * batch_size + batch_size]
            batch_y = Y_train[step * batch_size: step * batch_size + batch_size]
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
    acc_test = sess.run(accuracy, feed_dict={x: X_test[:1024],
                                        y: Y_test[:1024],
                                        keep_prob: 1.})
    print("Testing Accuracy:", acc_test)
    # Predict single images
    n_images = 4
    test_images = X_test[:n_images]
    preds = sess.run(pred_labels, feed_dict={x: test_images,
                                            keep_prob: 1.})
    # Display
    for i in range(n_images):
        plt.figure()
        plt.imshow(np.reshape(test_images[i], [256, 256]), cmap="gray")
        plt.show()
        print("Model prediction:", preds[i])

    train_writer.close()
    valid_writer.close()
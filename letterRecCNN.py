# Bowen Zheng
# 861130900
# Dec-9-2016
# CS 229
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import optparse 
  
############################
#####Processing data
############################  

imageLength = 128
imageWidth = 16
imageHeight = 8
numClasses = 26
LEARNING_RATE = 1e-3

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--train", action="store_true", default=False, help="Train the neural network, should also specify data with --datafile=FILE")
    optParser.add_option("--test", action="store_true", default=False, help="Test the neural network with input data --datafile=FILE")
    optParser.add_option("--crossValid", action="store_true", default=False, help="10-fold corss validation with input data --datafile=FILE")
    optParser.add_option("--datafile", action="store", dest="datafile", default='handwriting.data', type="string", help="set input data file")
    options, args = optParser.parse_args()
    return options

def display(img):
    #image = img.values.reshape(imageWidth, imageHeight)
    image = np.reshape(img, (imageWidth, imageHeight))
    plt.axis('off')
    plt.imshow(image, cmap=cm.binary)
    plt.savefig('image.png')
    

def convertLabelToVector(labels, numClasses):
    numLabels = labels.shape[0]
    labelsVector = np.zeros((numLabels, numClasses))
    labelsVector = labelsVector.astype(np.uint8)
    for i in range(numLabels):
        labelsVector[i, labels[i]] = 1
    return labelsVector
    
def shuffle(data, perm):
    data_shuffle = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        perm_i = perm[i]
        data_shuffle[i:i+1] = data[perm_i:perm_i+1]
    return data_shuffle
    
# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')    


def training(train_images, train_labels, epoch_len):
    ############################
    #####TensorFlow Graph
    ############################ 
    x = tf.placeholder(tf.float32, shape=[None, imageLength])
    y = tf.placeholder(tf.float32, shape=[None, numClasses])


    #conv_layer_1
    feature1 = 32
    W_hidden1 = weight_variable([5, 5, 1, feature1])
    b_hidden1 = bias_variable([feature1])
    
    x_image = tf.reshape(x, [-1, imageWidth, imageHeight, 1])

    h_hidden1 = tf.nn.relu(conv2d(x_image, W_hidden1) + b_hidden1)
    h_pool1 = max_pool_2x2(h_hidden1)

    #conv_layer_2
    feature2 = 64
    W_hidden2 = weight_variable([5, 5, feature1, feature2])
    b_hidden2 = bias_variable([feature2])

    h_hidden2 = tf.nn.relu(conv2d(h_pool1, W_hidden2) + b_hidden2)
    h_pool2 = max_pool_2x2(h_hidden2)
    
    #densely connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 2 * feature2])
    W_fc1 = weight_variable([4 * 2 * feature2, imageLength])
    b_fc1 = bias_variable([imageLength])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #dropout
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #out_layer
    W_fc2 = weight_variable([imageLength, numClasses])
    b_fc2 = bias_variable([numClasses])
    

    #fx = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    fx = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # cost function
    fx_0_1 = tf.nn.softmax(fx)
    cross_entropy = tf.reduce_mean(-y * tf.log(fx_0_1))
    
    correct_prediction = tf.equal(tf.argmax(fx_0_1,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # optimisation function
    #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    
    batch_size = 100
    data_size = train_images.shape[0]
    for epoch in range(epoch_len):
        for batch_i in range(data_size/batch_size):
            batch_images = train_images[batch_i*batch_size : (batch_i+1)*batch_size]
            batch_labels = train_labels[batch_i*batch_size : (batch_i+1)*batch_size] 
            sess.run(train_step, feed_dict = {x: batch_images, y: batch_labels, keep_prob: 1.0})
        perm = np.arange(data_size)
        np.random.shuffle(perm)
        train_images = shuffle(train_images, perm)
        train_labels = shuffle(train_labels, perm)

    
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./CNNmodel/model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()
    
    
    
def testing(test_images, test_labels):
    x = tf.placeholder(tf.float32, shape=[None, imageLength])
    y = tf.placeholder(tf.float32, shape=[None, numClasses])


    #conv_layer_1
    feature1 = 32
    W_hidden1 = weight_variable([5, 5, 1, feature1])
    b_hidden1 = bias_variable([feature1])
    
    x_image = tf.reshape(x, [-1, imageWidth, imageHeight, 1])

    h_hidden1 = tf.nn.relu(conv2d(x_image, W_hidden1) + b_hidden1)
    h_pool1 = max_pool_2x2(h_hidden1)

    #conv_layer_2
    feature2 = 64
    W_hidden2 = weight_variable([5, 5, feature1, feature2])
    b_hidden2 = bias_variable([feature2])

    h_hidden2 = tf.nn.relu(conv2d(h_pool1, W_hidden2) + b_hidden2)
    h_pool2 = max_pool_2x2(h_hidden2)
    
    #densely connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 2 * feature2])
    W_fc1 = weight_variable([4 * 2 * feature2, imageLength])
    b_fc1 = bias_variable([imageLength])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #dropout
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #out_layer
    W_fc2 = weight_variable([imageLength, numClasses])
    b_fc2 = bias_variable([numClasses])
    

    fx = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    sess = tf.InteractiveSession()
    new_saver = tf.train.Saver()
    new_saver.restore(sess, "./CNNmodel/model.ckpt")
    print("model restored.")
    
    correct_prediction = tf.equal(tf.argmax(fx,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #sess.run(accuracy, feed_dict={x: test_images,y: test_labels})
    prediction = tf.argmax(fx,1)

    predicted_labels = prediction.eval(feed_dict = {x: test_images, keep_prob: 1.0})
    accuracy = accuracy.eval(feed_dict={x: test_images,y: test_labels, keep_prob: 1.0})
    
    sess.close()

    return accuracy, predicted_labels
    

def crossValidation(images, labels, epoch_len):
    Nfold = 10
    SIZEfold = images.shape[0] / Nfold
    train_accuracy = []
    test_accuracy = []
    
    #for i in range(Nfold):
    for i in range(Nfold):
        idx = np.arange(i*SIZEfold, (i+1)*SIZEfold)
        train_images = np.delete(images, idx, 0)
        train_labels = np.delete(labels, idx, 0)
        
        test_images = images[idx]
        test_labels = labels[idx]
        
        ############################
        #####TensorFlow Graph
        ############################
        x = tf.placeholder(tf.float32, shape=[None, imageLength])
        y = tf.placeholder(tf.float32, shape=[None, numClasses])


        #conv_layer_1
        feature1 = 32
        W_hidden1 = weight_variable([5, 5, 1, feature1])
        b_hidden1 = bias_variable([feature1])
        
        x_image = tf.reshape(x, [-1, imageWidth, imageHeight, 1])
    
        h_hidden1 = tf.nn.relu(conv2d(x_image, W_hidden1) + b_hidden1)
        h_pool1 = max_pool_2x2(h_hidden1)
    
        #conv_layer_2
        feature2 = 64
        W_hidden2 = weight_variable([5, 5, feature1, feature2])
        b_hidden2 = bias_variable([feature2])
    
        h_hidden2 = tf.nn.relu(conv2d(h_pool1, W_hidden2) + b_hidden2)
        h_pool2 = max_pool_2x2(h_hidden2)
        
        #densely connected layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 2 * feature2])
        W_fc1 = weight_variable([4 * 2 * feature2, imageLength])
        b_fc1 = bias_variable([imageLength])
        
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        #dropout
        keep_prob = tf.placeholder('float')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        #out_layer
        W_fc2 = weight_variable([imageLength, numClasses])
        b_fc2 = bias_variable([numClasses])
        

        #fx = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        fx = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


        # cost function
        fx_0_1 = tf.nn.softmax(fx)
        cross_entropy = tf.reduce_mean(-y * tf.log(fx_0_1))

        # optimisation function
        #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(fx_0_1,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.InteractiveSession()

        sess.run(tf.global_variables_initializer())
    
        batch_size = 100
        data_size = train_images.shape[0]
        for epoch in range(epoch_len):
            for batch_i in range(data_size/batch_size):
                batch_images = train_images[batch_i*batch_size : (batch_i+1)*batch_size]
                batch_labels = train_labels[batch_i*batch_size : (batch_i+1)*batch_size] 
                sess.run(train_step, feed_dict = {x: batch_images, y: batch_labels, keep_prob: 1.0})
            perm = np.arange(data_size)
            np.random.shuffle(perm)
            train_images = shuffle(train_images, perm)
            train_labels = shuffle(train_labels, perm)
            ta = accuracy.eval(feed_dict={x: train_images,y: train_labels, keep_prob: 1.0})
            va = accuracy.eval(feed_dict={x: test_images,y: test_labels, keep_prob: 1.0})
            print('train_acc = %.2f, validation_acc = %.2f' % (ta, va))

        train_accuracy.append(accuracy.eval(feed_dict={x: train_images,y: train_labels, keep_prob: 1.0}))
        test_accuracy.append(accuracy.eval(feed_dict={x: test_images,y: test_labels, keep_prob: 1.0}))
        sess.close()       
    trainA =  np.mean(train_accuracy)
    testA = np.mean(test_accuracy)
    
    return trainA, testA
        
    
    
if __name__ == "__main__":
    options = get_options()
    print options
    
    datafile = options.datafile
    print 'datafile is ' + datafile
    
    data = pd.read_csv(datafile, sep=' ', header=None)
    
    images = data.iloc[:,2:]
    labels = data.iloc[:,0]
    
    images = images.as_matrix()
    labelsVec = convertLabelToVector(labels, numClasses)
    
    if options.train:    
        training(images, labelsVec, 10)

    if options.crossValid:
        trainA, testA = crossValidation(images, labelsVec, 10)
        print('cross_validation: train_accuracy=%.4f, validation_accuracy=%.4f' % (trainA, testA))
 
    if options.test:
        accuracy, predicted_labels = testing(images, labelsVec)
        np.savetxt('predicted_labels.txt', predicted_labels, fmt='%i')
        print 'predicted_labels = ' + str(predicted_labels) + ".\n For details see predicted_labels.txt"
        print 'test_accuracy = ' + str(accuracy)

      

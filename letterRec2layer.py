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


def training(train_images, train_labels, epoch_len):
    ############################
    #####TensorFlow Graph
    ############################ 
    x = tf.placeholder(tf.float32, shape=[None, imageLength])
    y = tf.placeholder(tf.float32, shape=[None, numClasses])

    #hidden_layer_1
    hidden_units_1 = 30
    W_hidden1 = weight_variable([imageLength, hidden_units_1])
    b_hidden1 = bias_variable([hidden_units_1])
    
    h_hidden1 = tf.nn.relu(tf.matmul(x, W_hidden1) + b_hidden1)
    
    #hidden_layer_2
    hidden_units_2 = 30
    W_hidden2 = weight_variable([hidden_units_1, hidden_units_2])
    b_hidden2 = bias_variable([hidden_units_2])
    
    h_hidden2 = tf.nn.relu(tf.matmul(h_hidden1, W_hidden2) + b_hidden2)
    
    #output_layer
    W = weight_variable([hidden_units_2, numClasses])
    b = bias_variable([numClasses])

    fx = tf.matmul(h_hidden2, W) + b


    # cost function
    fx_0_1 = tf.nn.softmax(fx)
    cross_entropy = tf.reduce_mean(-y * tf.log(fx_0_1))

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
            sess.run(train_step, feed_dict = {x: batch_images, y: batch_labels})
        perm = np.arange(data_size)
        np.random.shuffle(perm)
        train_images = shuffle(train_images, perm)
        train_labels = shuffle(train_labels, perm)

    
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()
    
    
    
def testing(test_images, test_labels):
    x = tf.placeholder(tf.float32, shape=[None, imageLength])
    y = tf.placeholder(tf.float32, shape=[None, numClasses])

    #hidden_layer_1
    hidden_units_1 = 30
    W_hidden1 = weight_variable([imageLength, hidden_units_1])
    b_hidden1 = bias_variable([hidden_units_1])
    
    h_hidden1 = tf.nn.relu(tf.matmul(x, W_hidden1) + b_hidden1)
    
    #hidden_layer_2
    hidden_units_2 = 30
    W_hidden2 = weight_variable([hidden_units_1, hidden_units_2])
    b_hidden2 = bias_variable([hidden_units_2])
    
    h_hidden2 = tf.nn.relu(tf.matmul(h_hidden1, W_hidden2) + b_hidden2)
    
    #output_layer
    W = weight_variable([hidden_units_2, numClasses])
    b = bias_variable([numClasses])

    fx = tf.matmul(h_hidden2, W) + b
    
    sess = tf.InteractiveSession()
    new_saver = tf.train.Saver()
    new_saver.restore(sess, "./model/model.ckpt")
    print("model restored.")
    
    correct_prediction = tf.equal(tf.argmax(fx,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #sess.run(accuracy, feed_dict={x: test_images,y: test_labels})
    prediction = tf.argmax(fx,1)

    predicted_labels = prediction.eval(feed_dict = {x: test_images})
    accuracy = accuracy.eval(feed_dict={x: test_images,y: test_labels})
    
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


        #hidden_layer_1
        hidden_units_1 = 30
        W_hidden1 = weight_variable([imageLength, hidden_units_1])
        b_hidden1 = bias_variable([hidden_units_1])
    
        h_hidden1 = tf.nn.relu(tf.matmul(x, W_hidden1) + b_hidden1)
    
        #hidden_layer_2
        hidden_units_2 = 30
        W_hidden2 = weight_variable([hidden_units_1, hidden_units_2])
        b_hidden2 = bias_variable([hidden_units_2])
    
        h_hidden2 = tf.nn.relu(tf.matmul(h_hidden1, W_hidden2) + b_hidden2)
    
        #output_layer
        W = weight_variable([hidden_units_2, numClasses])
        b = bias_variable([numClasses])

        #fx = tf.nn.relu(tf.matmul(h_hidden2, W) + b)
        fx = tf.matmul(h_hidden2, W) + b


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
                sess.run(train_step, feed_dict = {x: batch_images, y: batch_labels})
            perm = np.arange(data_size)
            np.random.shuffle(perm)
            train_images = shuffle(train_images, perm)
            train_labels = shuffle(train_labels, perm)
            ta = accuracy.eval(feed_dict={x: train_images,y: train_labels})
            va = accuracy.eval(feed_dict={x: test_images,y: test_labels})
            print('epoch = %i, train_acc = %.2f, validation_acc = %.2f' % (epoch, ta, va))

        train_accuracy.append(accuracy.eval(feed_dict={x: train_images,y: train_labels}))
        test_accuracy.append(accuracy.eval(feed_dict={x: test_images,y: test_labels}))
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
        training(images, labelsVec, 20)

    if options.crossValid:
        trainA, testA = crossValidation(images, labelsVec, 20)
        print('cross_validation: train_accuracy=%.4f, validation_accuracy=%.4f' % (trainA, testA))
 
    if options.test:
        accuracy, predicted_labels = testing(images, labelsVec)
        np.savetxt('predicted_labels.txt', predicted_labels, fmt='%i')
        print 'predicted_labels = ' + str(predicted_labels) + ".\n For details see predicted_labels.txt"
        print 'test_accuracy = ' + str(accuracy)

      

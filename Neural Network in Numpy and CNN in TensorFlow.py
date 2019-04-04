import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import string
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

# ################################PREAMBLE#########################################################
# Code optimizes  notMNIST dataset using two methods: a MLP Neural Net using Numpy and a powerful #
#  Convolutional Neural Network with TensorFlow. When the script is run in the interpreter, the   #
#  output will be the training, validation, and test accuracy and loss vectors for each method    #
# #################################################################################################


def neuralNumpy():
###################################################################################
#                 SUMMARY OF FINAL DATA (NEURAL NUMPY)                            #
# LAYER   TIME(s)   TRAIN LOSS/ACCURACY     VAL LOSS/ACCURACY  TEST LOSS/ACCURACY #
# 100      27.56       0.142640/96.21       0.362000/89.83      0.388943/89.90    #
# 500     129.44       0.129509/96.72       0.354060/90.37      0.383086/90.05    #
# 1000    232.41       0.122815/96.93       0.345485/90.32      0.377778/90.35    #
# 2000    426.98       0.121036/96.82       0.338872/90.67      0.367983/90.38    #
###################################################################################
# IMPORTANT: We removed a separate bias vector and incorporated it into the weight#
#            matrix by adding a row to the weight matrix and a column of ones to  #
#            any data vector coming in. When we update the weights in backwards   #
#            propogation, we ensure we only update the weights and not the biases #
#            by indexing correctly, or so we think.                               #
###################################################################################

    start1 = time.time() # Start Timer
    W1, W2, v1, v2 = globalVariableDefine() # Retrieve weight and v initilizations

    fig, (ax1, ax2) = plt.subplots(2, 1) # Open a 2x1 figure for live update plot

    for i in range(Numpy_Epochs):
        # Calculate and Store Training, Validation, and Test Error and Accuracy
        x2, x1, s1, x0, W2 = forwardPass(trainData, W1, W2)
        store_training_error[i], store_training_accuracy[i] = CE(trainTarget, x2)

        x2Temp, x1Temp, s1Temp, x0Temp, W2Temp = forwardPass(validData, W1, W2)
        store_valid_error[i], store_valid_accuracy[i] = CE(validTarget, x2Temp)

        x2Temp, x1Temp, s1Temp, x0Temp, W2Temp = forwardPass(testData, W1, W2)
        store_test_error[i], store_test_accuracy[i] = CE(testTarget, x2Temp)

        # Calculate new gradients using backwards propogation
        W1grad, W2grad = backwardPass(trainTarget, x2, x1, s1, x0, W2)

        # Update bias terms. This is done wihout momentum as the problem set had the update rule for the weights only
        W1[0,:] = W1[0,:] - learn_rate*W1grad[0,:]
        W2[0,:] = W2[0,:] - learn_rate*W2grad[0,:]

        # Update momentum terms
        v1 = gamma*v1 + learn_rate*W1grad[1:,:]
        v2 = gamma*v2 + learn_rate*W2grad[1:,:]

        # Update weights
        W1[1:,:] = W1[1:,:] - v1
        W2[1:,:] = W2[1:,:] - v2

        # Live update of accuracy and loss graphs
        ax1.cla()
        ax1.plot(store_training_error[0:i])
        ax1.plot(store_valid_error[0:i])
        ax1.plot(store_test_error[0:i])

        ax2.cla()
        ax2.plot(store_training_accuracy[0:i])
        ax2.plot(store_valid_accuracy[0:i])
        ax2.plot(store_test_accuracy[0:i])
        plt.pause(0.05)

    Loss_and_Acc_Numpy = np.array([store_training_error, store_valid_error, store_test_error, store_training_accuracy, store_valid_accuracy, store_test_accuracy])
    print('Time Taken to Train, Validate, and Test Numpy Neural = ', np.around(time.time() - start1,2)) # End timer

    return Loss_and_Acc_Numpy

def forwardPass(x, W1,W2):
    x0 = np.append(np.ones((x.shape[0],1)),x, axis = 1) # Append column of ones to first layer for the bias terms
    s1 = computeLayer(x0,W1)
    x1 = relu(s1) # Calculate ReLU
    x1 = np.append(np.ones((x.shape[0],1)),x1, axis = 1) # Append column of ones to hidden layer for the bias terms
    s2 = computeLayer(x1,W2)
    x2 = softmax(s2) # Probability Labels
    return x2, x1, s1, x0, W2

def backwardPass(y, x2, x1, s1, x0, W2):
    delta2 = np.subtract(x2,y) # Resultant of using gradCE. Full analytical derivation of gradCE can be found in our report!
    gradW2 = np.matmul(np.transpose(x1),delta2) # Math
    delta1 = np.multiply(np.matmul(delta2,np.transpose(W2[1:,:])),reluPrime(s1)) # More Math
    gradW1 = np.matmul(np.transpose(x0),delta1) # Even more Math. Derivations in the report!
    return gradW1, gradW2

def relu(x):
   return np.maximum(x, 0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x,axis=1,keepdims=True))
    return e_x / e_x.sum(axis=1,keepdims=True)

def computeLayer(X, W):
    return np.matmul(X,W)

def CE(target, prediction):
    accuracy = np.mean(np.argmax(target,axis = 1) ==  np.argmax(prediction,axis=1))*100
    loss =  np.multiply(-1/target.shape[0],np.trace(np.matmul(np.log(prediction),np.transpose(target))))
    return loss, accuracy

def reluPrime(x): #Derivative of ReLU
     x[x<=0] = 0
     x[x>0] = 1
     return x


def playGame(x,y, W1, W2):  # For Neural Net Numpy only! Send the data, the labels, and W1, and W2 and it will make predictions for you!
    dictionary = dict(enumerate(string.ascii_uppercase, 0))
    for i in range(x.shape[0]):
        x2, x1, s1, x0, W2 = forwardPass(x[i,None], W1,W2)
        predict = np.asscalar(np.argmax(x2,axis=1))
        plt.figure()
        plt.cla()
        plt.imshow(np.array(x[i]).reshape(28,28), cmap = 'gray') # Show each row as a picture
        plt.show()
        print("The system has predicted ", dictionary[predict])
        time.sleep(2) # Wait 2 seconds



def globalVariableDefine():
    #Define for Numpy Section
    global trainData, validData, testData, trainTarget, validTarget, testTarget, W1, W2, v1, v2, Numpy_Epochs, gamma, learn_rate, store_training_error, store_valid_error, store_test_error, store_training_accuracy, store_valid_accuracy, store_test_accuracy
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    Numpy_Epochs = 200 + 1 # Need to add one as 200 epochs means 201 periods
    layer0Units = 784  # Number of pixels in the image
    layer1Units = 1000 # Change this for playing around with hidden layer size
    layer2Units = 10 # Classes to class
    W1 = np.random.normal(0,2/(layer0Units+layer1Units),(layer0Units+1,layer1Units)) # Mr. Xavier initilization. Note +1 for including bias term
    W2 = np.random.normal(0,2/(layer1Units+layer2Units),(layer1Units+1,layer2Units))
    v1 = np.full((layer0Units,layer1Units),1e-5) # Mr/Madam TA recommended initilization
    v2 = np.full((layer1Units,layer2Units),1e-5)
    gamma = 0.95
    learn_rate = 1e-5 # Gives smoothest output to loss and accuracy graphs

    #Bunch of stroge holders for all the errors and accuracy values we will calculate
    store_training_error = np.zeros((Numpy_Epochs,1))
    store_valid_error= np.zeros((Numpy_Epochs,1))
    store_test_error= np.zeros((Numpy_Epochs,1))
    store_training_accuracy = np.zeros((Numpy_Epochs,1))
    store_valid_accuracy = np.zeros((Numpy_Epochs,1))
    store_test_accuracy = np.zeros((Numpy_Epochs,1))
    return W1, W2, v1, v2

def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

        trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    return reshapeData(trainData), reshapeData(validData), reshapeData(testData),trainTarget, validTarget, testTarget

def reshapeData(dataset):
    img_h = img_w = 28             # MNIST images are 28x28
    global img_size_flat
    img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
    dataset = dataset.reshape((-1,img_size_flat)).astype(np.float64)
    return dataset

def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def buildGraph():
    alpha = 1e-4 # Set learning rate
    tf.reset_default_graph() # Clear any previous junk
    tf.set_random_seed(421)

    labels = tf.placeholder(shape=(None, 10), dtype='int32')
    reg = tf.placeholder(tf.float32,None, name='regulaizer')
    isTraining = tf.placeholder(tf.bool)

    # Initialize Weights and Biases. When Mr. Xavier's initializor is set to uniform = False, a normal distribution is used
    weights = {
    'kernel': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
    'w1': tf.get_variable('W1', shape=( 14*14*32,784), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
    'w2': tf.get_variable('W2', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
}
    biases = {
    'b0': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
    'b1': tf.get_variable('B1', shape=(784), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
    'b2': tf.get_variable('B2', shape=(10), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
}
    # Step 1 - Input Layer
    trainingInput = tf.placeholder(tf.float32, shape=(None, 28, 28,1))

    # Step 2 - Convolutional Layer
    conv1 = tf.nn.conv2d(trainingInput, weights['kernel'], strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['b0'])

    # Step 3 - ReLU Activation
    x = tf.nn.relu(conv1)

    # Step 4 - Batch Normalization Layer
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
    xNorm = tf.nn.batch_normalization(x,mean,variance,None,None,1e-5)

    # Step 5 - Pooling Layer - Stride length not specified for this layer in the assignment so used what most people use on the world-wide web
    pool = tf.nn.max_pool(xNorm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Step 6 - Flatten layer
    poolFlatten  =tf.reshape(pool, [-1,14*14*32])

    # Step 7 - Fully Connected Layer and Dropout. Dropout done as an "if" statement only for training dataset.
    #          To change keep porbability, change the second number in rate
    layer_784 = tf.nn.bias_add(tf.matmul(poolFlatten, weights['w1']), biases['b1'])
    toReLU = tf.cond(isTraining, lambda: tf.nn.dropout(layer_784, rate = 1.0 - 1.0), lambda: layer_784)

    # Step 8 - ReLU Activation
    reluOutput = tf.nn.relu(toReLU)

    # Step 9 - Fully Connected Layer
    predict = tf.nn.bias_add(tf.matmul(reluOutput, weights['w2']), biases['b2'])

    # Step 10 - SoftMax Output
    outputClass = tf.argmax(tf.nn.softmax(predict), axis=1)

    # Step 11 - Cross Entropy Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict)) +reg*(tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']))

    # Step 12 - Calculate Prediction Accuracy
    correct_prediction = tf.equal(outputClass, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100

    # Step 13 - Define ADAM Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)

    return optimizer, loss, trainingInput, labels, reg, accuracy, isTraining

def CNN():
####################################################################################
#                SUMMARY OF FINAL DATA (CNN - NO VARYING)                          #
#         TIME(s)   TRAIN LOSS/ACCURACY   VAL LOSS/ACCURACY  TEST LOSS/ACCURACY    #
#          69.2       7.286e-6/100         0.550950/92.55        0.636015/92.40    #
####################################################################################
#                SUMMARY OF FINAL DATA (CNN - VARY LAMBDA)                         #
#  L      TIME(s)   TRAIN LOSS/ACCURACY   VAL LOSS/ACCURACY  TEST LOSS/ACCURACY    #
# 0.01     71.9       0.117795/100         0.370324/92.92       0.391423/92.55     #
# 0.1      70.7       0.421761/93.75       0.507754/92.70       0.505923/92.99     #
# 0.5      70.6       1.063455/93.75       0.969677/89.82       0.960630/90.31     #
####################################################################################
#                SUMMARY OF FINAL DATA (CNN - VARY KEEP RATE)                      #
#  P      TIME(s)   TRAIN LOSS/ACCURACY   VAL LOSS/ACCURACY  TEST LOSS/ACCURACY    #
# 0.5      71.5        4.560e-6/100        0.535601/92.93       0.588981/92.62     #
# 0.75     71.0        5.111e-6/100        0.583457/92.67       0.667736/91.18     #
# 0.9      70.8        1.196e-5/100        0.567780/92.75       0.656130/92.51     #
####################################################################################

    start1 = time.time() # Start Timer
    optimizer, loss, trainingInput, labels, reg, accuracy, isTraining = buildGraph()
    batch_size = 32
    SGD_training_epochs = 50+1 # Need to add one as 50 epochs means 51 periods
    validData4D = np.reshape(validData, (-1,28,28,1)) # Reshpae validation data
    testData4D = np.reshape(testData, (-1,28,28,1))   # Reshape test data


    #-----------------------Initialize Storage Variables-----------------------#
    store_training_error = np.zeros((SGD_training_epochs,1))
    store_valid_error= np.zeros((SGD_training_epochs,1))
    store_test_error= np.zeros((SGD_training_epochs,1))
    store_training_accuracy = np.zeros((SGD_training_epochs,1))
    store_valid_accuracy = np.zeros((SGD_training_epochs,1))
    store_test_accuracy = np.zeros((SGD_training_epochs,1))

    init = tf.global_variables_initializer()   # Initialize session
    fig, (ax1, ax2) = plt.subplots(2, 1) # Open a 2x1 figure for live update plot

    with tf.Session() as sess:
        sess.run(init)

        batch_number = int(trainTarget.shape[0]/batch_size) # Calculate batch number

        for i in range(SGD_training_epochs): # Loop across SGD_training_epochs
            trainDat, trainTar = shuffle(trainData,trainTarget) # Shuffle data every epoch using shuffle function provided
            trainData4D = np.reshape(trainDat, (-1,1,1,1)) # Reshape to a 4D matrix to feed into CNN

            batch_index = np.random.permutation(trainData4D.shape[0]) # Reshuffle training data
            ySplit = np.split(trainTarget[batch_index],batch_number) # Split into the number of batches
            xSplit = np.split(trainData[batch_index],batch_number) # Split into the number of batches

            for j in range(len(xSplit)): # Loop through each batch
                # Let us OPTIMIZE! Set isTraining to True to enable dropout for training only. Change reg if you want to have regularization.
                _, store_training_error[i], store_training_accuracy[i] = sess.run([optimizer,loss,accuracy], feed_dict = {trainingInput: xSplit, labels: ySplit, reg: 0.0, isTraining: True})

            # Calculate and store validation and test error and accuracy. Change reg if you want to have regularization.
            store_valid_error[i], store_valid_accuracy[i]= sess.run([loss,accuracy], feed_dict = {trainingInput: validData4D, labels: validTarget, reg: 0.0, isTraining: False})
            store_test_error[i], store_test_accuracy[i] = sess.run([loss, accuracy], feed_dict = {trainingInput: testData4D, labels: testTarget, reg: 0.0,  isTraining: False})

            # Live update of accuracy and loss graphs
            ax1.cla()
            ax1.plot(store_training_error[0:i])
            ax1.plot(store_valid_error[0:i])
            ax1.plot(store_test_error[0:i])

            ax2.cla()
            ax2.plot(store_training_accuracy[0:i])
            ax2.plot(store_valid_accuracy[0:i])
            ax2.plot(store_test_accuracy[0:i])
            plt.pause(0.05)


        Loss_and_Acc_adam = np.array([store_training_error, store_valid_error, store_test_error, store_training_accuracy, store_valid_accuracy, store_test_accuracy])

        print('Time Taken to Train, Validate, and Test TensorFlow Neural = ', np.around(time.time() - start1,2))
        sess.close()
        return Loss_and_Acc_adam

# Plotting Function
def plotting_time(a,b,c,d):
    plt.figure(1)
    plt.clf()
    plt.plot(a[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(a[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(a[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('CNN: No Dropout or Regularization Loss Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    plt.ylim(0)


    plt.figure(2)
    plt.clf()
    plt.plot(a[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(a[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(a[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('CNN: No Dropout or Regularization Accuracy Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylim(90,101)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()


    plt.figure(3)
    plt.clf()
    plt.plot(b[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(b[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(b[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('CNN: $\lambda = 0.01$ Loss Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()


    plt.figure(4)
    plt.clf()
    plt.plot(b[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(b[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(b[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('CNN: $\lambda = 0.01$ Accuracy Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(85,101)
    plt.grid(which = 'both', axis = 'both')
    plt.show()


    plt.figure(5)
    plt.clf()
    plt.plot(c[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(c[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(c[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('CNN: $\lambda = 0.1$ Loss Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()


    plt.figure(6)
    plt.clf()
    plt.plot(c[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(c[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(c[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('CNN: $\lambda = 0.1$ Accuracy Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(85,95)
    plt.grid(which = 'both', axis = 'both')
    plt.show()


    plt.figure(7)
    plt.clf()
    plt.plot(d[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(d[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(d[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('CNN: $\lambda = 0.5$ Loss Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()


    plt.figure(8)
    plt.clf()
    plt.plot(d[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(d[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(d[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('CNN: $\lambda = 0.5$ Accuracy Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylim(85,95)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()

    plt.figure(9)
    plt.clf()
    plt.plot(b[2,:], 'b-', label=r'$\lambda = 0.01$', linewidth = 4, color = 'blue')
    plt.plot(c[2,:], 'r-', label=r'$\lambda = 0.1$', linewidth = 4, color = 'orange')
    plt.plot(d[2,:], 'r-', label=r'$\lambda = 0.5$', linewidth = 4, color = 'green')
    plt.title('CNN: Compare Different Regularization Losses on Test Set', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()


    plt.figure(10)
    plt.clf()
    plt.plot(b[5,:], 'k-', label=r'$\lambda = 0.01$', linewidth = 4, color = 'blue')
    plt.plot(c[5,:], 'b-', label=r'$\lambda = 0.1$', linewidth = 4, color = 'orange')
    plt.plot(d[5,:], 'r-', label=r'$\lambda = 0.5$', linewidth = 4, color = 'green')
    plt.title('CNN: Compare Different Regularization Accuracies on Test Set', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.ylim(85,95)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()



if __name__ == "__main__":
    globalVariableDefine()
    print("Running Neural Numpy.....\n")
    lossAccuracyNumpy = neuralNumpy()
    print("\nComplete!")
    print("\nRunning CNN.....\n")
    lossAccurcyCNN = CNN()
    print("\nComplete! Bye Bye!")

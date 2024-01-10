# Nakka, Tulasi
# 1001_928_971
# 2023_10_29
# Assignment_03_01

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.regularizers import l2

def confusion_matrix(y_true, y_pred, n_classes=10):
    #
    # Compute the confusion matrix for a set of predictions
    #
    # the shape of the confusion matrix should be (n_classes, n_classes)
    # The (i, j)th entry should be the number of times an example with true label i was predicted label j
    # Only use numpy.
    # Do not use any libraries to use this function (e.g. sklearn.metrics.confusion_matrix, or tensorflow.math.confusion_matrix, ..)
     # Compute the confusion matrix for a set of predictions
    conf_matrix = np.zeros((n_classes, n_classes))
    for i in range(len(y_pred)):
        true_class = int(y_true[i])
        pred_class = int(y_pred[i])
        conf_matrix[true_class, pred_class] += 1
    return conf_matrix



def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):

    tf.keras.utils.set_random_seed(5368) # do not remove this line
    model = tf.keras.models.Sequential()

    # All layers that have weights should have L2 regularization with a regularization strength of 0.0001 (only use kernel regularizer)
    # imported keras.regulaizer above
    
    # - Convolutional layer with 8 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(8, (3,3), (1,1), 'same', activation='relu', kernel_regularizer=l2(0.0001), input_shape=X_train[0].shape))
    
    # - Convolutional layer with 16 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(16,(3,3), (1,1), 'same', activation='relu', kernel_regularizer=l2(0.0001)))
   
    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    model.add(MaxPooling2D((2,2), (2,2)))

    # - Convolutional layer with 32 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(32, (3,3), (1,1), 'same', activation='relu', kernel_regularizer=l2(0.0001)))
    
    # - Convolutional layer with 64 filters, kernel size 3 by 3 , stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(64, (3,3), (1,1), 'same', activation='relu', kernel_regularizer=l2(0.0001)))

    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    model.add(MaxPooling2D((2,2), (2,2)))
    
    # - Flatten layer
    model.add(Flatten())
    
    # - Dense layer with 512 units and ReLU activation
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.0001)))
    
    # - Dense layer with 10 units with linear activation
    model.add(Dense(10, activation='linear', kernel_regularizer=l2(0.0001)))
    
    # - a softmax layer
    model.add(tf.keras.layers.Activation("softmax"))

    # The neural network should be trained using the Adam optimizer with default parameters. The loss function should be categorical cross-entropy.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # The number of epochs should be given by the 'epochs' parameter. The batch size should be given by the 'batch_size' parameter.
    #history requires validation_split to pass testcase 'test_accuracy_on_mnist' as validation set automatically generates from train_set and it was mentioned 80% accuarcy so validation_split=0.2 and for other values the testcase is failing
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # predicting model with test data

    y_pred = model.predict(X_test)
    y_pred = np.array(np.argmax(y_pred, axis=1))

    # You should save the keras model to a file called 'model.h5' (do not submit this file). When we test run your program we will check "model.h5"
    #save(model)
    model.save('model.h5')

    # You should compute the confusion matrix on the test set and return it as a numpy array.
    Y_test_conf = np.argmax(Y_test, axis=1)
    matrix = confusion_matrix(Y_test_conf, y_pred, Y_test.shape[1])

    # You should plot the confusion matrix using the matplotlib function matshow (as heat map) and save it to 'confusion_matrix.png'
    plot_confusion_matrix(matrix)

    # You will return a list with the following items in the order specified below:
    # - the trained model
    # - the training history (the result of model.fit as an object)
    # - the confusion matrix as numpy array
    # - the output of the model on the test set (the result of model.predict) as numpy array

    return model, history, matrix, y_pred

# reference for plotting confusion matrix: https://vitalflux.com/python-draw-confusion-matrix-matplotlib/
def plot_confusion_matrix(confusion_matrix, save_path='confusion_matrix.png'):
   
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size as needed
    ax.matshow(confusion_matrix, cmap="hsv",alpha=0.3)

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            number = str(int(confusion_matrix[i, j]))
            ax.text(j, i, number, va='center', ha='center', size='xx-large')

    set_labels_and_ticks(confusion_matrix)
    draw_grid_lines(confusion_matrix)

    plt.savefig(save_path)
    plt.close()

def set_labels_and_ticks(confusion_matrix):
    plt.xticks(np.arange(confusion_matrix.shape[1]))
    plt.yticks(np.arange(confusion_matrix.shape[0]))
    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predictions', fontsize=18)
    plt.title('Confusion Matrix', fontsize=20)

def draw_grid_lines(confusion_matrix):
    for i in range(confusion_matrix.shape[0] - 1):
        plt.axhline(i + 0.5, color='red', linewidth=1.5)
        plt.axvline(i + 0.5, color='red', linewidth=1.5)

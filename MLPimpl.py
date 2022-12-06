#IMPORTS

from __future__ import print_function, division
import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator
from func import SoftmaxFunc, ReLU, CrossEntropyFunc, accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


    #   Classifier Implementation
class MLP():

    def __init__(self, hidden_layer, iterations=300, learningRate=0.00001): 
        self.hidden_layer = hidden_layer
        self.iterations = iterations
        self.learningRate = learningRate

        self.activation_func = ReLU()
        self.output_activation = SoftmaxFunc()
        self.loss_func = CrossEntropyFunc()

    def _initialize_weights(self, X, y): # Set Weights
        _, data_features = X.shape
        s, output_val = y.shape
       
        max_val   = 1 / math.sqrt(data_features)  # Hidden layer
        self.W  = np.random.uniform(-max_val, max_val, (data_features, self.hidden_layer))
        self.w0 = np.zeros((1, self.hidden_layer))
        
        max_val   = 1 / math.sqrt(self.hidden_layer) # Output layer
        self.V  = np.random.uniform(-max_val, max_val, (self.hidden_layer, output_val))
        self.v0 = np.zeros((1, output_val))

    def fit(self, X, y):

        self._initialize_weights(X, y)

        for i in range(self.iterations):
            input_val = X.dot(self.W) + self.w0
            output_val = self.activation_func(input_val)

            # Layer - Output
            output_layer_input = output_val.dot(self.V) + self.v0
            y_pred_val = self.output_activation(output_layer_input)

            gradient_wrt_out_l_input = self.loss_func.gradient(y, y_pred_val) * self.output_activation.gradient(output_layer_input)
            gradient_v = output_val.T.dot(gradient_wrt_out_l_input)
            gradient_v0 = np.sum(gradient_wrt_out_l_input, axis=0, keepdims=True)

            # Layers - Hidden 
            gradient_wrt_hidden_l_input = gradient_wrt_out_l_input.dot(self.V.T) * self.activation_func.gradient(input_val)
            gradient_w = X.T.dot(gradient_wrt_hidden_l_input)
            gradient_w0 = np.sum(gradient_wrt_hidden_l_input, axis=0, keepdims=True)

            # Update weights - w0, W, v0, V
            self.V  -= self.learningRate * gradient_v
            self.v0 -= self.learningRate * gradient_v0
            self.W  -= self.learningRate * gradient_w
            self.w0 -= self.learningRate * gradient_w0
    
    # Make predictions based on model
    def predict(self, X): 
        input_val = X.dot(self.W) + self.w0
        output_val = self.activation_func(input_val)
        output_layer_input = output_val.dot(self.V) + self.v0
        y_pred_val = self.output_activation(output_layer_input)
        return y_pred_val
    
    def classify(self, X):
        y_pred_val = self.predict(X)
        
        return y_pred_val.idxmax(), y_pred_val.max()

    # Weights values
    def dumpWeights(self):
        np.savetxt('w0.txt',self.w0,fmt='%.2f')
        np.savetxt('v0.txt',self.v0,fmt='%.2f')
        np.savetxt('W.txt',self.W,fmt='%.2f')
        np.savetxt('V.txt',self.V,fmt='%.2f')

# Main function
def main():
    path_loc = "sample8" #Dataset path name
    batches = ImageDataGenerator().flow_from_directory(directory=path_loc, target_size=(64,64), batch_size=10000)

    imgs, labels = next(batches)
    print(imgs.shape)

    img = np.reshape(imgs, (imgs.shape[0], -1)) #Reshape image to fit implementation

    X_train, X_rem, y_train, y_rem = train_test_split(img/255.,labels, train_size=0.8)
    X_val, X_test, y_val, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

    # Shape of data
    print(f"Shape of train set is {X_train.shape}")
    print(f"Shape of test set is {X_test.shape}")
    print(f"Shape of train label is {y_train.shape}")
    print(f"Shape of test labels is {y_test.shape}")
    print(f"Shape of val set is {X_val.shape}")
    print(f"Shape of val labels is {y_val.shape}")

    # Classfier - MLP
    clf = MLP(hidden_layer=256, iterations=300, learningRate=0.00001)

    start = time.time()

    clf.fit(X_train, y_train)

    end = time.time()
    print(end - start, 's')

    y_pred_val = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    y_val_pred = np.argmax(clf.predict(X_val), axis=1)
    y_val = np.argmax(y_val, axis=1)

    y_tr_pred = np.argmax(clf.predict(X_train), axis=1)
    y_train = np.argmax(y_train, axis=1)

    accuracy_test = accuracy(y_test, y_pred_val)
    val_accuracy = accuracy(y_val, y_val_pred)
    accuracy_tr = accuracy(y_train, y_tr_pred)

    print ("Training Accuracy:", accuracy_tr)
    print ("Testing Accuracy:", accuracy_test)
    print ("Validation Accuracy:", val_accuracy)

    # View weights value
    clf.dumpWeights()

    # Creates confusion matrix and display result
    p = confusion_matrix(y_test, y_pred_val)
    c =  confusion_matrix(y_test, y_pred_val)
    np.savetxt('confusionMatrix',c,fmt='%.0f')
    print(c)

    # Classification Report containing recall, precision and f1-score
    report = classification_report(y_test, y_pred_val)
    print(report)

    # confusion matrix plot
    cm_display = ConfusionMatrixDisplay(c).plot()
    cm_display.plot()
    plt.show()
    

if __name__ == "__main__":
    main()
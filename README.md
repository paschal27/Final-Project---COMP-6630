# COMP-6630--Final-Project--Malware Classification with Multi-Layer Perceptron

## Step-by-step procedure
Implementation can be executed on any python IDE like VsCode or Pycharm.

Download code from provided github link, and load it into any IDE of your choice.

All files including the dataset must be located in the same folder.

Path name must be set to the name of the dataset in order to aviod any errors.

Click on run, and the required acurracy output should be shown in the terminal.

Also, a confusion matrix plot and a classification report file will be generated.


## Dataset Usage

Malimg dataset can be downloaded through this link: old.vision.ece.ucsb.edu/spam/malimg.shtml.

Details about the Malimg dataset (original paper) can be found here: https://dl.acm.org/doi/10.1145/2016904.2016908

The Malimg subset (8 classes) used in the proposed model can be extracted from the complete

## Proposed Multi-Layer Perceptron Model

Our proposed multi-layer perceptron model consists of 1 input layer of ```64*64=4096``` neurons, 1 hidden layer of ```256``` neurons, and 1 output layer of either ```8``` or ```19``` neurons. An output layer of ```8``` classes is used when training from the Malimg subset (8 classes, as indicated in the original Malimg paper). An output layer of ```19``` classes is used for classifying the complete Malimg with merging of malware variants of the same type.

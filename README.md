# classificatio_small_data
In this repo there is a solution to an image classification problem using a small amount of data. 
The dataset of images is ciFAIR-10 which consists of 10 classes: the training set is made of 50 training images per class and the test set of 1000 images per class. 
Out of the 50 images per class that can be used during training, 20 are used solely for validation and the remaining for training.
The solution is a Convolutional Neural Networks based on SqueezeNet, a network that has been created with the aim of reducing the parameters of a CNN while maintaining competitive accuracy. However, some changes have been made to the basic structure.
A more extensive description of the solution can be found in the pdf file in the `report` folder.

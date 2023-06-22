# Mask-Detection-Project
Keras/Tensorflow based mask detection project

The project has 4 main files/folders:
1. the PortableModel folder holds a pretrained AI, so you don't need to retrain it
2. the CamToQRToDetection_CompletePackage_UsesRevivialMachine.py tests the model
3. reducedTrainerSize creates, trains, and saves the model
4. Dataset holds the directory structure to test the model effectively

# Function

The model testing file tests multiple things after it takes a photo using the computer webcam. 
First, it checks for a QR code in the image indicating an exception to the mask rule. The product was originally designed for use in a hospital setting where exceptional circumstances may occur. 
Second, it predicts how likely a mask is being worn in the photo. The higher the outputted number, the lower the chance of a mask-bearing individual.
Third, it goes through a SmallerSet containing sample images of users with and without masks to test its accuracy

# Model Structure

I used a Convolutional Neural Network to process the images. Essentially, I used convolutional and max pooling layers until the result could be processed by a regular neural network. From there, I flattened it so it can be processed by the dense layers. A few dropout layers were added to this portion to minimize overfitting
See the attached folder CNN_Structure.png for a visual, specific look into how it is constructed.

# Results

A test on a dataset separate from the one used to train it was used. When tested, I were able to achieve a 97% accuracy, with the majority of the mistakes being false positives. However, it does not detect whether an individual is improperly wearing a mask. Insufficient testing was done on this project to consider it actually viable, but the product mostly functions nonetheless.


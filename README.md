# Face Recognition using Siamese Networks
A face recognition model developed using computer vision and Tensorflow

##Introduction
This code is an implementation of a face recognition system using Siamese networks. The system is trained on the LFW (Labeled Faces in the Wild) dataset, which contains more than 13,000 images of faces collected from the web. The goal of the system is to be able to recognize a person's face in an image and match it to a known person in the dataset.

##Methods
The system uses a Siamese network architecture, which consists of two identical neural networks, each taking one of the two input images. The output of the two networks is then compared using a contrastive loss function. This allows the network to learn a compact representation of the face that is invariant to changes in lighting, pose, and other variations.

The code is implemented using the Keras library with a Tensorflow backend. The model is trained for 20 epochs with a batch size of 128. The Adam optimizer is used for training with a learning rate of 0.0001.

##Testing
To test the trained model, we can use the predict() function to feed in an image of a face and get the output of the model. The output will be a list of distances between the input image and the images in the dataset. The image that has the smallest distance to the input image is considered to be the match. If the distance is above a certain threshold, the image is not recognized.

To test the model on new images, we can use the following code snippet:

'''
from keras.models import load_model
import numpy as np

### Load the trained model
model = load_model('siamese_net.h5')

### Load the test image
test_image = ...

### Normalize the image
test_image = test_image / 255.

### Reshape the image
test_image = np.reshape(test_image, (1, 128, 128, 3))

### Predict
pred = model.predict(test_image)

### Threshold
threshold = ...

### Get the index of the image with the smallest distance
best_match_index = np.argmin(pred)

### Get the name of the person
names = {0: 'Name1', 1: 'Name2', ...}

if pred[0][0] < threshold:
    print('Image not recognized')
else:
    print('The person in the image is', names[best_match_index])
'''

##Labels Mapping
To map the labels to the respective string names of the persons in the dataset, you can use a dictionary. For example, assuming that the names of the persons are stored in a list called person_names, you can create a dictionary that maps the labels to the names as follows:

'''
names = {i: person_names[i] for i in range(len(person_names))}
'''

##Conclusion
This code provides a simple example of how to build a face recognition system using Siamese networks. The system can be further improved by using more advanced architectures, such as Triplet Networks, and by fine-tuning the hyperparameters. Additionally, the system can be trained on a larger dataset to improve its performance. Overall, the Siamese network is a powerful tool for face recognition and can be applied to other related tasks such as face verification and clustering.

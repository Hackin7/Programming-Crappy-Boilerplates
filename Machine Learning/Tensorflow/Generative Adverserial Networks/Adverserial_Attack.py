# https://github.com/PacktPublishing/Neural-Networks-with-Keras-Cookbook/blob/master/Chapter08/Adversarial_attack.ipynb
# https://www.tensorflow.org/tutorials/generative/adversarial_fgsm


import os
os.system("cd /tmp && wget https://www.dropbox.com/s/eqadwd8m21sf0lg/cat.JPG && cd -")
### Preprocessing Image #########################################################
import cv2
import numpy as np
# Preprocess it so it can be passed into an inception network
def preprocess(path):
    img = cv2.imread(path)#'/content/cat.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299,299))
    original_image = cv2.resize(img,(299,299)).astype(float)
    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.
    original_image = np.expand_dims(original_image, axis=0)
    return img, original_image
img, original_image = preprocess('/tmp/cat.JPG')


import matplotlib.pyplot as plt
#%matplotlib inline
#plt.imshow(img)
#plt.axis('off')

### Using Inception ############################################################
from keras.preprocessing import image
from keras.applications import inception_v3

def inception_predict(image):
    model = inception_v3.InceptionV3()
    predictions = model.predict(image)
    return inception_v3.decode_predictions(predictions)

model = inception_v3.InceptionV3()
predictions = model.predict(original_image)
predicted_classes = inception_v3.decode_predictions(predictions, top=1)
imagenet_id, name, confidence = predicted_classes[0][0]
print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))
print(inception_v3.decode_predictions(predictions))

### Adverserial Attack #########################################################
'''
Make a Persian cat be detected as an Elephant
1. Define a loss function
    * The loss tis the probability of the image belonging to the African elephant class
    * The higher the loss, the closer are we to our objective -> Maximising our loss function
2. Calculate the gradient of change in the loss with respect to the change in the input
    * This step helps in understanding the input pixels that move the output towards our objective
3. Update the input image based on the calculated gradients
    * Ensure that the pixel values in the original image is not translated by more than 3 pixels in the final image
    * This ensures that the resulting image is humanly indistinguishable from the original image
4. Repeat steps 2 and step 3 until predicted as African elephant
'''
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image

# https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta/66222183
tf.compat.v1.disable_eager_execution()

# Load pre-trained image recognition model
model = inception_v3.InceptionV3()
# Grab a reference to the first and last layer of the neural net
model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

# Limits of change
max_change_above = np.copy(original_image) + 0.01
max_change_below = np.copy(original_image) - 0.01
hacked_image = np.copy(original_image)

### Parameters #############
learning_rate = 10000#0.1
object_type_to_fake = 386 # Prediction Vector Index (African elephant is 386)

cost_function = model_output_layer[0, object_type_to_fake] # Index the output layer
gradient_function = K.gradients(cost_function, model_input_layer)[0]
grab_cost_and_gradients_from_model = K.function([model_input_layer], [cost_function, gradient_function])
cost = 0.0
# https://stackoverflow.com/questions/48142181/whats-the-purpose-of-keras-backend-function

### Attack #####################
counter = 0
while cost < 0.80: # no. loops is number of epoches
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
    if cost < 10**(-2): learning_rate = 10000
    else: learning_rate = 10
    hacked_image += gradients * learning_rate
    hacked_image = np.clip(hacked_image, max_change_below, max_change_above) #Boundaries
    counter += 1
    print(counter, cost)

print(inception_predict(hacked_image))

### Deprocessing hacked image
hacked_image = hacked_image/2
hacked_image = hacked_image + 0.5
hacked_image = hacked_image*255
hacked_image = np.clip(hacked_image, 0, 255).astype('uint8')

cv2.imwrite("/tmp/output.jpg", hacked_image)

### Comparison of images ##########################################
plt.subplot(131)
plt.imshow(img)
plt.title('Original image')
plt.axis('off')
plt.subplot(132)
plt.imshow(hacked_image[0,:,:,:])
plt.title('Hacked image')
plt.axis('off')
plt.subplot(133)
plt.imshow(img - hacked_image[0,:,:,:])
plt.title('Difference')
plt.axis('off')

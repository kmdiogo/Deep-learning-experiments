from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Load the json file that contains the model's structure
f = Path("fashion_mnist_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("fashion_mnist_weights.h5")

# Load an image file to test, resizing it to 32x32 pixels (as required by this model)
#img = image.load_img("shoe.png", target_size=(28, 28))
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
img = x_test[1]

# Convert the image to a numpy array
#image_to_test = image.img_to_array(img)
#image_to_test = img.transpose(2,0,1).reshape(3,-1)
# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
#list_of_images = np.expand_dims(image_to_test, axis=0)

# Make a prediction using the model
results = model.predict(img)

# Since we are only testing one image, we only need to check the first result
single_result = results[0]

# We will get a likelihood score for all 10 possible classes. Find out which class had the highest score.
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

# Get the name of the most likely class
class_label = class_labels[most_likely_class_index]

# Print the result
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))
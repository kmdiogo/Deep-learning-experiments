from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Load the json file that contains the model's structure
f = Path("mnist_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# load the model's trained weights
model.load_weights("mnist_weights.h5")

#img = Image.open('number8.jpg').convert('L').resize((28,28))
# Load handwritten image and resize to 28x28
img = image.load_img('number8.jpg', target_size=(28, 28))

# Convert the image to a numpy array
image_to_test = image.img_to_array(img)
plt.imshow(image_to_test)
plt.show()
image_to_test = image_to_test[:,:,0]
flat = image_to_test.ravel()
image_to_test = flat



#np.reshape(image_to_test, (28, 28))

# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
list_of_images = np.expand_dims(image_to_test, axis=0)

# Make a prediction using the model
results = model.predict(list_of_images)

# Since we are only testing one image, we only need to check the first result
single_result = results[0]

# We will get a likelihood score for all 10 possible classes. Find out which class had the highest score.
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]


# Print the result
print("This is image is a {} - Likelihood: {:2f}".format(most_likely_class_index, class_likelihood))
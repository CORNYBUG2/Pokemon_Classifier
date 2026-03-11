import tensorflow as tf
import numpy as np
from keras.preprocessing import image

model = tf.keras.models.load_model("pokemon_class.keras")

class_names = ['Bulbasaur','Charizard','Pikachu','Snorlax','Squirtle']

img = image.load_img("image.png", target_size=(224,224))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

pred = model.predict(img_array)

print(class_names[np.argmax(pred)])
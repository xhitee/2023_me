from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D

num_class = 4

#basemodel = ResNet50V2(weights='imagenet', include_top=False, input_shape=(112, 112, 3), classifier_activation="softmax")
#model = Sequential()
#model.add(basemodel)
#model.add(Flatten())
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(units, activation='softmax'))
basemodel = ResNet50V2(weights='imagenet', include_top=False, input_shape=(112, 112, 3), classifier_activation="softmax")
model = Sequential()
model.add(basemodel)
model.add(GlobalAveragePooling2D())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_class, activation='softmax'))

##################################model road issuse..#########################
# model.load('/home/white/Desktop/vision/augmented_ResNet50_Epoch100.h5')
model = load_model('/home/white/Desktop/vision/ResNet50_Epoch100.h5')
##############################################################################

# model.predict('/home/white/Desktop/vision/white.jpeg')
image_path = '/home/white/Desktop/vision/kim.jpeg'

image = Image.open(image_path)
image = image.resize((112, 112))
image = np.array(image)
image = np.expand_dims(image, axis = 0)

prediction = model.predict(image)

print(prediction)



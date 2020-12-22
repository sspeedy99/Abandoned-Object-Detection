import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import model_from_json
import json


with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_weights.h5')

test_data_dir = 'Test_dir/'

img_width, img_height = 224, 224
batch_size=2

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_data = test_datagen.flow_from_directory(
test_data_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='categorical')

test_data.reset()


yhat = model.predict_generator(test_data,  verbose=1)
y_classes = yhat.argmax(axis=-1)
val = {0: '1', 1: '2', 2: '3'}

final = y_classes[0]

print("The threat classification is: ", val[final])
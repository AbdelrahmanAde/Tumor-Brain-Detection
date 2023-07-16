import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.applications.resnet50 import preprocess_input

class ImageProcessor:
    def __init__(self):
        with open('models\grading\label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)

    def crop(self, image):
        # Get the indices of the pixels containing the brain
        brain_indices = np.where(image > 10)

        # Get the minimum and maximum indices
        min_x, max_x = np.min(brain_indices[0]), np.max(brain_indices[0])
        min_y, max_y = np.min(brain_indices[1]), np.max(brain_indices[1])

        # Crop the image
        image = image[min_x:max_x, min_y:max_y]

        return image

    def process_data(self, my_dict):
        new_data = my_dict

        encoded_data = {}
        for col, val in new_data.items():
            if col in self.label_encoders:
                encoded_data[col] = self.label_encoders[col].transform([val])[0]
            else:
                encoded_data[col] = val
        test = np.array(list(encoded_data.values())).reshape(1, -1)
        return test

    def process_img(self, image_path):
        IMG_SIZE = (224, 224)

        img = cv2.imread(image_path)
        img = self.crop(img)
        img_resized = cv2.resize(img, IMG_SIZE)

        # Convert the image to a NumPy array
        img_array = np.array(img_resized)
        x_process = preprocess_input(img_array)
        x = np.array(x_process)
        x = np.expand_dims(x, axis=0)
        return x

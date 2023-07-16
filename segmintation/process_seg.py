import tensorflow.keras.backend as K
import tensorflow as tf
import cv2
import numpy as np

class coff:
    def __init__(self, smooth=1.0):
        self.smooth = smooth
    
    def dice_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred)
        return (2.0 * intersection + self.smooth) / (union + self.smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def bce_dice_loss(self, y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return self.dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)

    def iou(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        sum_ = K.sum(y_true + y_pred)
        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)
        return jac

class ImageProcessor2:
    def __init__(self):
        pass

    def preprocess_image(self, image_path):
        # Load image and resize it to (256, 256)
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(256, 256))

        # Normalize image values to [0, 1]
        image = image / 255.0

        # Add batch dimension to the image
        image_ex = np.expand_dims(image, axis=0)

        return image_ex

    def merging(self, image, pred):
        p=np.reshape(pred,(256,256,1)) 
        thresh=cv2.threshold(p,0.002,1,cv2.THRESH_BINARY)
        thresh2=np.uint8(thresh[1])
        # Resize mask to match image size
        image=cv2.resize(image,dsize=(256,256))

        image=image /255.0

        mask = cv2.resize(thresh2, (image.shape[1], image.shape[0]))

        # Create green color segmentation
        seg = np.zeros_like(image)
        seg[:, :, 1] = mask

        # Merge image and segmentation
        merged = cv2.addWeighted(image, 0.8, seg, 0.4, 0)
        merged = (merged * 255).astype('uint8')
        merg = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

        return merg

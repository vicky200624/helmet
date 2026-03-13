import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
import zipfile
import pathlib
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
import cv2
from tensorflow.keras import layers, Model

# Load the saved model without recompiling to avoid deserializing legacy metric objects
model = tf.keras.models.load_model(
    "C:\\Users\\vicky\\Downloads\\helmet\\helmet_detector.h5",
    compile=False,
)

# If a dedicated test image isn't provided, fall back to the first image from the training dataset
# (this helps the script run out-of-the-box).
test_image_path = "C:/Users/vicky/Downloads/helmet/test.jpg"

if not os.path.isfile(test_image_path):
    dataset_folder = "C:/Users/vicky/Downloads/detection.v1i.tensorflow/train"
    possible = [f for f in os.listdir(dataset_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not possible:
        raise FileNotFoundError(
            f"No test image found at {test_image_path} and no images found in {dataset_folder}."
        )

    test_image_path = os.path.join(dataset_folder, possible[0])
    print(f"No test.jpg found; using {test_image_path} from the training folder.")

img = cv2.imread(test_image_path)
if img is None:
    raise ValueError(f"Unable to read image: {test_image_path}")

h, w = img.shape[:2]

resized = cv2.resize(img,(224,224))
input_img = resized/255.0
input_img = np.expand_dims(input_img,axis=0)

bbox_pred,class_pred = model.predict(input_img)

x,y,bw,bh = bbox_pred[0]

xmin = int((x-bw/2)*w)
ymin = int((y-bh/2)*h)
xmax = int((x+bw/2)*w)
ymax = int((y+bh/2)*h)

label = np.argmax(class_pred)

text = "Helmet" if label==0 else "No Helmet"

cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
cv2.putText(img,text,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

cv2.imshow("Helmet Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
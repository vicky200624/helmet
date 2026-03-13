import os
import csv
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SIZE = 224

train_path = "C:/Users/vicky/Downloads/detection.v1i.tensorflow/train"
val_path = "C:/Users/vicky/Downloads/detection.v1i.tensorflow/valid"
test_path = "C:/Users/vicky/Downloads/detection.v1i.tensorflow/test"


def load_dataset(path):

    images = []
    bbox = []
    classes = []

    # Preferred structure (Roboflow export uses a flat folder + _annotations.csv)
    annotation_csv = os.path.join(path, "_annotations.csv")
    image_dir = path

    if os.path.isfile(annotation_csv):
        # Read annotations into a dict: filename -> list of rows
        annotations = defaultdict(list)
        with open(annotation_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations[row["filename"]].append(row)

        for file in os.listdir(image_dir):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            if file not in annotations:
                # No annotations for this image
                continue

            ann = annotations[file][0]
            w_img = float(ann.get("width", img.shape[1]))
            h_img = float(ann.get("height", img.shape[0]))
            xmin = float(ann["xmin"])
            ymin = float(ann["ymin"])
            xmax = float(ann["xmax"])
            ymax = float(ann["ymax"])

            # Convert to normalized YOLO-style (x_center, y_center, w, h)
            x_center = ((xmin + xmax) / 2.0) / w_img
            y_center = ((ymin + ymax) / 2.0) / h_img
            w_box = (xmax - xmin) / w_img
            h_box = (ymax - ymin) / h_img

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            images.append(img)
            bbox.append([x_center, y_center, w_box, h_box])
            classes.append([1, 0])  # helmet class

        return np.array(images), np.array(bbox), np.array(classes)

    # Legacy structure: images/ + labels/ folder
    image_dir = os.path.join(path, "images")
    label_dir = os.path.join(path, "labels")

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError(
            f"Dataset directory not found. Expected structure:\n"
            f"  {path}/images\n"
            f"  {path}/labels\n"
            "Please set the correct `train_path`, `val_path`, and `test_path` values "
            "or ensure the dataset is downloaded and extracted."
        )

    for file in os.listdir(image_dir):

        img_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        with open(label_path, "r") as f:
            line = f.readline().split()

            cls = int(line[0])
            x = float(line[1])
            y = float(line[2])
            w = float(line[3])
            h = float(line[4])

        images.append(img)
        bbox.append([x, y, w, h])

        if cls == 0:
            classes.append([1, 0])
        else:
            classes.append([0, 1])
    return np.array(images), np.array(bbox), np.array(classes)


# Load datasets
X_train,y_bbox_train,y_class_train = load_dataset(train_path)
X_val,y_bbox_val,y_class_val = load_dataset(val_path)
X_test,y_bbox_test,y_class_test = load_dataset(test_path)

print("Train:",X_train.shape)
print("Validation:",X_val.shape)
print("Test:",X_test.shape)


# CNN Model
input_layer = tf.keras.Input(shape=(224,224,3))

x = layers.Conv2D(32,(3,3),activation='relu')(input_layer)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64,(3,3),activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128,(3,3),activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(256,activation='relu')(x)

bbox_output = layers.Dense(4,name="bbox")(x)
class_output = layers.Dense(2,activation="softmax",name="class")(x)

model = Model(inputs=input_layer,outputs=[bbox_output,class_output])

model.compile(
    optimizer="adam",
    loss={
        "bbox":"mse",
        "class":"categorical_crossentropy"
    },
    metrics={
        "class":"accuracy"
    }
)

model.summary()


# Train
history = model.fit(
    X_train,
    {"bbox": y_bbox_train, "class": y_class_train},
    validation_data=(X_val, {"bbox": y_bbox_val, "class": y_class_val}),
    epochs=20,
    batch_size=8,
)

train_acc = history.history.get("class_accuracy", [None])[-1]
val_acc = history.history.get("val_class_accuracy", [None])[-1]
print(f"Final train class accuracy: {train_acc:.4f}")
print(f"Final val   class accuracy: {val_acc:.4f}")


# Test accuracy
results = model.evaluate(
    X_test,
    {"bbox": y_bbox_test, "class": y_class_test},
    verbose=2,
)

# Keras returns [total_loss, bbox_loss, class_loss, class_accuracy]
if isinstance(results, (list, tuple)) and len(results) >= 4:
    test_acc = results[-1]
    print(f"Test  class accuracy: {test_acc:.4f}")
else:
    print("Test results:", results)


# Save model
model.save("helmet_detector.h5")
print("Model saved")
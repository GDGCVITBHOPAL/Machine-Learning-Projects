# Importing Project Dependencies
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Setting up config for GPU training
if tf.test.is_gpu_available:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Loading in all the images and assigning target classes
def load_images(folder):
    imgs, targets = [], []
    for foldername in os.listdir(folder):
        loc = folder + "/" + foldername
        targets.append(len(os.listdir(loc)))
        for filename in os.listdir(loc):
            img = cv2.imread(os.path.join(loc, filename))
            img = cv2.resize(img, (86, 86))
            if img is not None and img.shape == (86, 86, 3):
                imgs.append(img)
        print(foldername)
    imgs = np.array(imgs)
    y = np.zeros(imgs.shape[0]).astype(int)
    j, n = 0, 0
    for i in targets:
        y[j:i + j] = n
        n += 1
        j = i + j
    return imgs, y


folder = "../input/drowsiness-detection"
X, y = load_images(folder)

# Splitting the data into 2 separate training and testing sets
def train_test_split(X, y, testing_size=0.2):
    no_of_rows = X.shape[0]
    no_of_test_rows = int(no_of_rows * testing_size)
    rand_row_num = np.random.randint(0, no_of_rows, no_of_test_rows)

    X_test = np.array([X[i] for i in rand_row_num])
    X_train = np.delete(X, rand_row_num, axis=0)

    y_test = np.array([y[i] for i in rand_row_num])
    y_train = np.delete(y, rand_row_num, axis=0)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = train_test_split(X, y, testing_size=0.2)
print(X_train[0].shape)

# Model building using sequential API
model = keras.Sequential(
        [
            keras.Input(shape=(86, 86, 3)),
            layers.Conv2D(75, 3, padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(5, 5)),
            layers.Conv2D(64, 2, padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, 3, padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax'),
        ]
    )
print(model.summary())

# Model compilation with keeping track of accuracy while training & evaluation process
model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

model.fit(X_train, y_train, batch_size=32, epochs=10)

model.evaluate(X_test, y_test, batch_size=32)

# Saving the model
model.save('my_model (1).h5')

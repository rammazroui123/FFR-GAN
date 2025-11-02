import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hides TensorFlow info/warning messages
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    keras.Input(shape=(5,)),            # define input explicitly
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])

model.summary()

import numpy as np
np.random.seed(1337)

from keras.layers import BatchNormalization, Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, save_model

model = Sequential((
    BatchNormalization(input_shape=[48, 160, 3]),
    Conv2D(16, 3, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(24, 3, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(36, 3, 3, activation='relu'),
    Conv2D(48, 2, 2, activation='relu'),
    Conv2D(48, 2, 2, activation='relu'),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense( 10, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', mtetrics=['accuracy'])

with open('./model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('./model-weights.h5')

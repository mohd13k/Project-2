#Project 2: Image Recognition
#Mohammad Khan
#500976279


#Step 1: Data Processing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Step 1: Data Processing

# Define input image shape
input_shape = (500, 500, 3)

# Define data directories
train_dir = '/Users/mohammadkhan/Desktop/AER850/Data/train'
validation_dir = '/Users/mohammadkhan/Desktop/AER850/Data/valid'
test_dir = '/Users/mohammadkhan/Desktop/AER850/Data/test'

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation Data Generator 
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Test Data Generator 
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(500, 500),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(500, 500),
    batch_size=32,
    class_mode='categorical'
)

# Step 2-3: Neural Network Architecture Design/Hyperparameter Analysis


model = Sequential()

# Layer 1 
model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), input_shape=(500, 500, 3)))
model.add(LeakyReLU(alpha=0.1))  
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2 
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1)))
model.add(LeakyReLU(alpha=0.1))  
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3 
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(units=64, activation='elu'))  
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary 
model.summary()

#Step 4: Model Evaluation
history = model.fit(train_generator,epochs=20,validation_data=validation_generator)
model.save("mohammad_model.h5")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

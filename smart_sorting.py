import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Define Paths and Hyperparameters
TRAIN_DIR = './dataset/dataset/train'
TEST_DIR = './dataset/dataset/test'
BATCH_SIZE = 32
IMG_SIZE = (224, 224) # Standard input size for MobileNetV2
EPOCHS = 2

# 2. Load and Preprocess the Dataset
print("Loading training data...")
train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

print("Loading test/validation data...")
validation_dataset = image_dataset_from_directory(
    TEST_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

class_names = train_dataset.class_names
print("Classes identified:", class_names)
NUM_CLASSES = len(class_names)

# Prefetching for performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# 3. Data Augmentation
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip('horizontal'),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
])

# 4. Build the Transfer Learning Model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False, 
    weights='imagenet'
)

# Freeze the base model 
base_model.trainable = False

# Create the final model architecture
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

# 5. Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# 6. Train the Model
print("Starting training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)

# 7. Evaluate the Model
loss, accuracy = model.evaluate(validation_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 8. Save the Model
model.save('smart_sorting_model.keras')
print("Model saved as 'smart_sorting_model.keras'")
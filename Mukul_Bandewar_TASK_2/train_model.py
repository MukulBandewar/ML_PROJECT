
import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

os.makedirs("model", exist_ok=True)

print("Loading EMNIST dataset...")

(train_data, test_data), info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

classes = info.features["label"].names

def prepare_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_data = train_data.map(
    prepare_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_data = test_data.map(
    prepare_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_data = train_data.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(128).prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training started...")

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=8
)

loss, accuracy = model.evaluate(test_data)

model.save("model/character_model.keras")

with open("model/classes.json", "w") as file:
    json.dump(classes, file)

print("Model saved successfully!")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


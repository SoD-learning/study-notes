# Install non-Google-colab version (https://www.tensorflow.org/install)
# pip install --upgrade pip
# pip install tensorflow

# Using this instead: https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf

# Get TensorFlow version
# print("TensorFlow version:", tf.__version__)  # 2.14.0

# Load a dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a machine learning model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Train and evaluate the model
model.fit(x_train, y_train, epochs=5)  # adjust model paras & minimise loss

model.evaluate(x_test, y_test, verbose=2)  # check performance on validation or test set

# output:
# Epoch 1/5
# 1875/1875 [==============================] - 3s 1ms/step - loss: 0.3003 - accuracy: 0.9135
# Epoch 2/5
# 1875/1875 [==============================] - 3s 1ms/step - loss: 0.1433 - accuracy: 0.9580
# Epoch 3/5
# 1875/1875 [==============================] - 3s 1ms/step - loss: 0.1103 - accuracy: 0.9667
# Epoch 4/5
# 1875/1875 [==============================] - 3s 1ms/step - loss: 0.0906 - accuracy: 0.9721
# Epoch 5/5
# 1875/1875 [==============================] - 3s 1ms/step - loss: 0.0740 - accuracy: 0.9771
# 313/313 - 0s - loss: 0.0760 - accuracy: 0.9762 - 338ms/epoch - 1ms/step
# 👉 accuracy is 98%

# # ! Not sure about this part, as nothing seems to be returned?
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model(x_test[:5])

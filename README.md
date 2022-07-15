# Tensorflow Deep Learning Notebooks
This repo code contain notebooks related to Deep learning.

# Table of contents
- [1- Keras Functional API](#1--Keras-Functional-API)
- [2- Adding Custom Loss Function](#2--Adding-Custom-Loss-Functions)

# 1- Keras Functional API

<details>
<summary>Click to expand!</summary>

#### 1- Keras Functional API
One great advantage of using the functional API is the additional flexibility in your model `architecture design`, where instead of each layer being linearly stacked in turn with other layers, you can have `branches`, `cycles`, `multiple inputs and outputs`, and a whole lot more.

<h3 align="center">Sequential API</h3>

```python
sequential_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                               tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                               tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

```
<h3 align="center">Functional API</h3>

```python
# input 
input_layer = tf.keras.Input(shape=(28, 28))

# hidden layer
flatten_layer = tf.keras.layers.Flatten()(input_layer)
first_dense = tf.keras.layers.Dense(10, activation=tf.nn.relu)(flatten_layer)
output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(first_dense)

# declare inputs and outputs
functional_model = Model(inputs=input_layer, outputs=output_layer)
```
[Link to Notebooks](https://github.com/devzohaib/Tensorflow_practice_notebooks/tree/master/1-%20Functional%20API%20Practice)

</details>

# 2- Adding Custom Loss Functions

<details>
<><summary>Click to expand!</summary>

#### 2- Adding Custom Loss Functions
To create a custom loss function, you'll need to create your own function that accepts two parameters , typically called `y_true` and `y_pred` as in prediction on these contain your true labels and your current predicted values. The loss will be some kind of a function that calculates the difference between the two.

```python
# loss function
def my_huber_loss(y_true, y_pred):
    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

# model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# adding loss function
model.compile(optimizer='sgd', loss=my_huber_loss)
# training
model.fit(X, Y, epochs=5)
```

[Link to Notebook](https://github.com/devzohaib/Tensorflow_practice_notebooks/tree/master/2-%20Custom%20Loss%20Functions)


</details>

# 2- Template heading for a section

<details>
<><summary>Click to expand!</summary>

#### 2- Template heading for a section

</details>

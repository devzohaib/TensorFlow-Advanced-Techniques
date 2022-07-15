# Tensorflow_practice_notebooks
this repo code contains notebooks related to Deep learning.

# Table of contents
- [Functioanl API](#1--Keras-Functional-API)

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
</details>

# 2- Adding Custom Loss Functions

<details>
<><summary>Click to expand!</summary>

#### 2- Adding Custom Loss Functions

</details>

# 2- Template heading for a section

<details>
<><summary>Click to expand!</summary>

#### 2- Template heading for a section

</details>
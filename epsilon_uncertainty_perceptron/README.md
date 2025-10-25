# Epsilon Uncertainty Perceptron (EUP)

> _"The earliest predecessors of modern `deep learning` were simple linear models motivated from a neuroscientific perspective. These models were designed to take a set of `n` input values, and associate them with an output `y`. These models would learn a set of weights and compute their output" — (Goodfellow, Bengio, & Courville, 2016, p. 14)_

## 1 Decomposition

1. **Inputs:** Each **simple perceptron** receives inputs `x` as a vector of values: 

$$
\mathbf{x} = (x_1, x_2, \dots, x_n)
$$

2.  **Weights:** For each input, the perceptron has assigned a weight `w`, that represents _"the importance"_ for the decision. The weights represent a different vector:
   
$$
\mathbf{w} = (w_1, w_2, \dots, w_n)
$$

4. **Linear Combination:** The first mathematical operation of a perceptron is a linear combination of inputs and weights:

$$
z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b
$$

4. **Bias:** Where bias `b` represents the _"additional displacement"_ that gives more flexibility to the data separation

5. **Regular Activation Function:** The value `z` is passed through a step function, which produces a binary output:

$$
h(x)=1 (x ≥ 0), 0 (x<0)
$$

6. **Epsilon Uncertainty Activation Function (EUAF):** Its similar to the step function, but add an epsilon margin (ε):

$$
h(x)=1 (x>ε), 0 (x<-ε), 0.5 (-ε ≤ x ≤ ε)
$$

7. **Learning Rule:** The perceptron learns by adjusting its weights with each training example `(x, y)`.

    **Where:**

    - `η` = learning rate `(0 < η ≤ 1)`
    - `y` = true label
    - `ŷ` = perceptron output

$$
w_i \leftarrow w_i + \eta (y - \hat{y}) x_i
$$

$$
b \leftarrow b + \eta (y - \hat{y})
$$

## 2. Epsilon Uncertainty Perceptron Framework

The EUP framework contains four main modules: `train`, `fine_tuning`, `inference`, and `load_model_into_cache`. The architecture of the framework is designed to rely on an `LRUCache`, which uses a least recently used cache called `ModelCache`. This design aims to implement a forced versioning system where multiple threads can handle various framework processes simultaneously while sharing the same cache (a memory locker was included to prevent race conditions).

### 2.1 Cache Usage `ModelCache`

To initialize a cache partition you need to define the model cache as a variable, the module `ModelCache` just expect the argument `cache_lengt: int`

```python
# initialize the shared perceptron cache
perceptron_cache = ModelCache(cache_length=7)
```

### 2.2 `SimplePerceptron` Framework: train, inference, fine tunning, load model

1. **Framework Initialization:** After the initialization of the `ModelCache` you need to initialize the framework. The argument that receive this object is the cache:

```python
# initialize the SimplePerceptron class
simple_perceptron = SimplePerceptron(perceptron_cache)
```

2. **Dataset Expected Format:** To train or fine-tune a new model, you need to structure the dataset in a expected format in a file-type `JSON` where you have a format `list[dict]` where one key of the dict is named "features" in a type of `list[float|int]` and other key named "label" whit a data type: `int` (0, 1). For example:

```json
[
    {"features": [0, 0], "label": 0},
    {"features": [0, 1], "label": 1},
    {"features": [1, 0], "label": 1},
    {"features": [1, 1], "label": 1}
]
```

3. **Train a new model:** To train a model, you need to use fe function `train` from the framework, te expected arguments are → `epochs: int`, `patience: int`, `labeled_dataset_path: str`, `learning_rate: float`, `model_metadata: dict[str, str]`. The expected format for the model metadata is a dict whit → `model_name: str`, `description: str`, `author: str`. The function return an ID of the model in cache.

    **Where:**

    - `epochs: int` → training loop iterations
    - `patience: int` → tolerance without improvement
    - `labeled_dataset_path: str` → dataset file path (relative)
    - `learning_rate: float` → weight update rate
    - `model_metadata: dict[str, str]` → model metadata dictionary
  
```python
# define the model metadata
model_metadata = {
    'model_name': "Simple Perceptron", 
    'description': "A simple perceptron trained with the gate `OR`", 
    'author': "Dylan Sutton Chavez"
}

# train the perceptron with specified parameters
cache_id = simple_perceptron.train(epochs=30, patience=3, labeled_dataset_path='gate-or.json', learning_rate=0.65, model_metadata=model_metadata)
```

And the expected output (a list of logs whit general information (`epoch`, `weights`, `bias`, `error`, and `time`) and indicators: `early stopping`, `model root`):

```bash
Epoch 1/30
    Weights: [0.5974724124598851, 0.5472716124598853] | Bias: 0.71221679 | Error: 0.75 | Time: 0.0112
Epoch 2/30
    Weights: [0.5974724124598851, 0.5472716124598853] | Bias: 0.71221679 | Error: 0.0 | Time: 0.0075
Epoch 3/30
    Weights: [0.5974724124598851, 0.5472716124598853] | Bias: 0.71221679 | Error: 0.0 | Time: 0.0078
Epoch 4/30
    Weights: [0.5974724124598851, 0.5472716124598853] | Bias: 0.71221679 | Error: 0.0 | Time: 0.00460001
Early Stopping
Model saved as `simple-perceptron.2025_10_15.json`
```

4. **Fine Tune a Core Model:** The fine tunning is very useful, because this allow the model to improve with the time. The name of the function is `fine_tuning`, and expect the next arguments → `epochs: int`, `patience: int`, `labeled_dataset_path: str`, `learning_rate: float`, `past_model_path: str`, `model_metadata: dict[str, str]`. The function return an ID of the model in cache.

    **Where:**

    - `past_model_path: str` → core model path

```python
# define the fine-tuned model metadata
model_metadata = {
    'model_name': "Simple Perceptron Tuned", 
    'description': "Fine-tuned simple perceptron using the gate `OR`", 
    'author': "Dylan Sutton Chavez"
}

# make fine-tuning to the past model
simple_perceptron.fine_tuning(epochs=10, patience=2, labeled_dataset_path='gate-or.json', learning_rate=0.65, past_model_path='simple-perceptron.2025_10_15.json', model_metadata=model_metadata)
```

Expected Ouput:

```bash
Epoch 1/10
    Weights: [0.5974724124598851, 0.5472716124598853] | Bias: 0.71221679 | Error: 0.0 | Time: 0.00640001
Epoch 2/10
    Weights: [0.5974724124598851, 0.5472716124598853] | Bias: 0.71221679 | Error: 0.0 | Time: 0.00789999
Early Stopping
Model saved as `simple-perceptron-tuned.2025_10_15.json`
```

> In this case we are using the same dataset, but we can use the set of your preference to make the fine-tuning (using the same scale, dimensions and coherent-set)

5. **Load Models in Memmory:** To load models in memmory, you can use the funcion `load_model_into_cache`. This function receive just the argument: `model_path: str`. Where `model_path` its a trained or fine tuned model. And this function returns the loaded ID of the model in cache.

```python
# load a saved model and make take de id
model_id = simple_perceptron.load_model_into_cache(model_path='simple-perceptron-tuned.2025_10_15.json')
```

6. **Inference whit Uncertainty (ϵ):** You can make inference to a created model, or to a fine-tuned model. To run inference you need to peak the module `inference`, that receive the arguments → `features: list[float|int]`, `cache_id: int`, `epsilon: float`.

    **Where:**

    - `features: list[float|int]` → input features vector
    - `cache_id: int` → recive an `ID` to find the `OBJECT` in the `CACHE`
    - `epsilon: float = None` → desviation (grey area) for the step function

```python
# make a prediction whit a model ID
prediction = simple_perceptron.inference(features=[0, 1], cache_id=cache_id)
print(prediction)
```

Expected Ouput (the model have that example in the dataset or a similar example):

```bash
1
```

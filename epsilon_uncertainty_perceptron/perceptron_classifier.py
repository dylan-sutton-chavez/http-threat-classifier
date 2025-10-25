from json import load, dump
from random import uniform
from time import strftime, localtime, perf_counter
from re import sub

from numpy import dot, array

from model_cache import ModelCache, TrainingInfo
from config import VECTOR_PROCESSING_SWITCH_POINT

class SimplePerceptron:
    def __init__(self, perceptron_cache: ModelCache):
        """
        parse a cache object to create a vertical-scalable architecture

        args:
            perceptron_cache: ModelCache → parse a created cache object

        output:
            None

        time complexity → o(1)
        """
        self.perceptron_cache: ModelCache = perceptron_cache

    def train(self, epochs: int, patience: int, labeled_dataset_path: str, learning_rate: float, model_metadata: dict[str, str]):
        """
        train perceptron model with early stopping and model saving

        args:
            epochs: int → training loop iterations
            patience: int → tolerance without improvement
            labeled_dataset_path: str → dataset file path
            learning_rate: float → weight update rate
            model_metadata: dict[str, str] → model metadata dictionary

        output:
            cache_id: int → an `ID` to find the `OBJECT` in the `CACHE`

        time complexity → o(e*n*f + n*f)
        """
        # load the dataset, and standarize the features
        dataset = self._load_json(file_path=labeled_dataset_path)
        standarized_dataset, means, standard_deviation = self._zscore_dataset(dataset)

        num_features = len(dataset[0]['features'])

        # initialize the weights and bias with random vaules
        weights = [round(uniform(-0.07, 0.07), 8) for _ in range(num_features)]
        bias = round(uniform(-0.07, 0.07), 8)

        training_info = TrainingInfo(epochs=epochs, learning_rate=learning_rate, dataset_path=labeled_dataset_path, model_metadata=model_metadata, patience=patience)
        cache_id = self.perceptron_cache.add_item(weights=weights, bias=bias, standard_deviation=standard_deviation, means=means, training_info=training_info)

        self._training_loop(standarized_dataset=standarized_dataset, cache_id=cache_id, model_metadata=model_metadata)

        return cache_id

    def fine_tuning(self, epochs: int, patience: int, labeled_dataset_path: str, learning_rate: float, past_model_path: str, model_metadata: dict[str, str]):
        """
        use a core pre-trained model, fine-tune with more data and save the model 

        args:
            epochs: int → training loop iterations
            patience: int → tolerance without improvement
            labeled_dataset_path: str → dataset file path
            learning_rate: float → weight update rate
            past_model_path: str → core model path
            model_metadata: dict[str. str] → model metadata dictionary

        output:
            cache_id: int → an `ID` to find the `OBJECT` in the `CACHE`

        time complexity → o(e*n*f)
        """
        past_model = self._load_json(past_model_path)
        dataset = self._load_json(labeled_dataset_path)

        # start the weights and bias with the loaded model
        weights = past_model['parameters']['weights']
        bias = past_model['parameters']['bias']

        # initialize the means and standar desviation to normalize the fatures
        means = past_model['normalization']['means']
        standard_deviation = past_model['normalization']['standard_deviation']

        standarized_dataset, _, _ = self._zscore_dataset(dataset, means, standard_deviation)

        training_info = TrainingInfo(epochs=epochs, learning_rate=learning_rate, dataset_path=labeled_dataset_path, model_metadata=model_metadata, patience=patience, past_model=past_model)
        cache_id = self.perceptron_cache.add_item(weights=weights, bias=bias, standard_deviation=standard_deviation, means=means, training_info=training_info)

        self._training_loop(standarized_dataset=standarized_dataset, cache_id=cache_id, past_model_path=past_model_path, model_metadata=model_metadata)
        
        return cache_id

    def inference(self, features: list[float], cache_id: int, epsilon: float = None):
        """
        performs prediction using a saved perceptorn model and input features

        args:
            features: list[float] → input features vector
            cache_id: int → recive an `ID` to find the `OBJECT` in the `CACHE`
            epsilon: float = None → desviation (grey area) for the step function

        output:
            float → prediction value (0, 1), or gray area (0.5)

        time complexity → o(f)
        """
        model = self.perceptron_cache.get_item(cache_id)

        # initialize the means and standar desviation to normalize the fatures
        means = model.means
        standar_desviation = model.standard_deviation

        # make a prediction and return the result
        y_pred = self._calculate_net_input(self._zscore(means, standar_desviation, features), cache_id)
        return self._apply_step_function(y_pred, epsilon)
    
    def load_model_into_cache(self, model_path: str):
        """
        model_path: str → path to the saved model file
        
        args:
            model_path: str → path to the saved model file
            
        output:
            cache_id: int → an `ID` to find the `OBJECT` in the `CACHE`
            
        time complexity → o(n)
        """
        model = self._load_json(model_path)
        
        return self.perceptron_cache.add_item(weights=model['parameters']['weights'], bias=model['parameters']['bias'], means=model['normalization']['means'], standard_deviation=model['normalization']['standard_deviation'])

    def _training_loop(self, standarized_dataset: list[dict], cache_id: int, model_metadata: dict[str, str], past_model_path: str = None):
        """
        training loop with number of epochs where train the model whit a given dataset

        args:
            standarized_dataset: list[dict] → calibrated dataset using z-score
            cache_id: int → recive an `ID` of a perceptron model
            model_metadata: dict(str, str) → a dictionary whit the full model metadata
            past_model_path: str = None → path for the past model

        output:
            None

        time complexity → o(e*n*f)
        """
        perceptron: object = self.perceptron_cache.get_item(cache_id)

        patience_counter: int = 0
        total_time: float = 0.0

        # iterate through each of the epochs
        for epoch in range(perceptron.training_info.epochs):
            elapsed_time, has_errors, errors = self._train_one_epoch(standarized_dataset, perceptron.training_info.learning_rate, cache_id)
            total_time += elapsed_time

            self._log_epoch_metrics(epoch, perceptron.training_info.epochs, errors, standarized_dataset, elapsed_time, cache_id)

            # early stopping logic, if the model makes no mistakes
            if has_errors:
                patience_counter: int = 0 # set to zero the `no-errors` counter

            else:
                patience_counter += 1 # increment one the `no-errors` counter
                if patience_counter >= perceptron.training_info.patience:
                    print(f"Early Stopping")

                    self._save_model(cache_id, total_time, model_metadata, past_model_path)
                    return
                
        # if the loop finish without early stopping, save the last epoch model
        self._save_model(cache_id, total_time, model_metadata, past_model_path)

    def _train_one_epoch(self, normalized_dataset: list[dict], learning_rate: float, cache_id: int):
        """
        execute one training epoch for the dataset

        args:
            normalized_dataset: list[dict] → labaled and normalized training dataset
            learning_rate: float → step size for weight updates
            cache_id: int → recive an `ID` of a perceptron model
        
        output:
            elapsed_time: float → duration of epoch execution 
            has_errors: bool → `True` if any misclassifications occurred
            errors: list → list of prediction errors per sample

        time complexity → o(n*f)
        """
        start = perf_counter()
        
        has_errors: bool = False
        errors: list[float] = []

        # loop through examples in the normalized dataset 
        for example in normalized_dataset:
            features = example['features']
            y_true = example['label']

            # make a prediction with the current model
            net_input = self._calculate_net_input(features, cache_id)
            y_pred = self._apply_step_function(net_input)

            error = y_true - y_pred

            # if the prediction is incorrect, apply the learning rule
            if error != 0:
                # adjust weights and bias of the model
                self._update_weights_and_bias(error, features, learning_rate, cache_id)

                has_errors: bool = True
                errors.append(error)

        elapsed_time = perf_counter() - start

        return elapsed_time, has_errors, errors

    def _log_epoch_metrics(self, epoch: int, epochs: int, errors: list[float], dataset: list[dict], elapsed_time: float, cache_id: int):
        """
        log epoch metric including `weights`, `bias`, `error`, and `time`

        args:
            epoch: int → current epoch index
            epochs: int → total training epochs
            errors: list[float] → misclassified samples in current epoch
            dataset: list[dict] → training dataset used
            elapsed_time: float → epoch execution time
            cache_id: int → recive an `ID` of a perceptron model

        return:
            None

        time complexity → o(1)
        """
        perceptron: object = self.perceptron_cache.get_item(cache_id)

        print(f"Epoch {epoch + 1}/{epochs}\n    Weights: {perceptron.weights} | Bias: {round(perceptron.bias, 8)} | Error: {len(errors) / len(dataset)} | Time: {round(elapsed_time * 1000, 8)}")

    def _save_model(self, cache_id: int, total_time: float, model_metadata: dict[str, str], past_model_path: str = None):
        """
        saves perceptron model, parameters, and training metadata to `JSON`

        args:
            cache_id: int → recive an `ID` of a perceptron model
            total_time: float → total training time in seconds
            model_metadata: dict(str, str) → model metadata
            past_model_path: str = None → name of the path for the mast model

        output:
            None

        time complexity → o(n)
        """
        perceptron: object = self.perceptron_cache.get_item(cache_id)

        model_dict = {
            "model_name": model_metadata['model_name'],
            "description": model_metadata['description'],
            "created_at": strftime("%Y-%m-%d %H:%M:%S", localtime()),
            "author": model_metadata['author'],

            "parameters": {
                "weights": perceptron.weights,
                "bias": perceptron.bias
            },

            "normalization": {
                "means": perceptron.means,
                "standard_deviation": perceptron.standard_deviation
            },

            "training": {
                "epochs": perceptron.training_info.epochs,
                "learning_rate": perceptron.training_info.learning_rate,
                "dataset": perceptron.training_info.dataset_path,
                "time": total_time * 1000
            }
        }

        # confirm if its reciving a past model, or if its a new model
        if past_model_path:
            past_model = perceptron.training_info.past_model

            model_dict.update({
                "pst_description": {
                    "name": past_model['model_name'],
                    "description": past_model['description'],
                    "created_at": past_model['created_at'],
                    "author": past_model['author'],
                    "pst_model_path": past_model_path
                },

                "pst_training": {
                    "epochs": past_model['training']['epochs'],
                    "learning_rate": past_model['training']['learning_rate'],
                    "dataset": past_model['training']['dataset'],
                    "time": past_model['training']['time']
                }
            })

        # clean the name and make a formated filename
        clean_name = lambda raw: sub(r'\s+', '-', raw.lower())
        model_filename = f"{clean_name(model_metadata['model_name'])}.{strftime(("%Y_%m_%d"), localtime())}.json"

        # save the model writing in a file
        with open(model_filename, 'w', encoding='utf-8') as model_file:
            dump(model_dict, model_file, indent=4) # save in `JSON` format
            print(f'Model saved as `{model_filename}`')

    def _calculate_net_input(self, input_features: list[float], cache_id: int):
        """
        compute the `weighted` sum of `features` plus `bias`

        args:
            input_features: list[float] → input feature vector
            cache_id: int → recive an `ID` of a perceptron model

        output:
            float → linear combination result

        time complexity →  o(f)

        maths:
            z = ∑ᵢ₌₁ⁿ wᵢ·xᵢ + b
        """
        perceptron: object = self.perceptron_cache.get_item(cache_id)

        # vectorized dot product for long vectors
        if len(input_features) >= VECTOR_PROCESSING_SWITCH_POINT:
            return dot(perceptron.weights, input_features) + perceptron.bias

        return sum(w * x for w, x in zip(perceptron.weights, input_features)) + perceptron.bias
    
    def _update_weights_and_bias(self, prediction_error: float, features: list[float], learning_rate: float, cache_id: int):
        """
        updates perceptron `weights` and `bias` based on prediction `error`

        args:
            prediction_error: float → difference between `True` and predicted label
            features: list[float] → input feature vector
            learning_rate: float → step size for `weight` updates
            cache_id: int → recive an `ID` of a perceptron model

        output:
            None

        time complexity → o(f)

        maths:
            b ← b + η · (y - ŷ)
            wᵢ ← wᵢ + η · (y - ŷ) xᵢ
        """
        perceptron: object = self.perceptron_cache.get_item(cache_id)
        
        bias: float = perceptron.bias + (learning_rate * prediction_error)
        adjustment_scalar: float = learning_rate * prediction_error
                
        # vectorized dot product for long vectors
        if len(features) >= VECTOR_PROCESSING_SWITCH_POINT:
            # convert the current wieghts and features to an array
            current_weights = array(perceptron.weights)
            features_array = array(features)

            weights: list[float] = (current_weights + adjustment_scalar * features_array).tolist()
        
        # python normal operation more eficient for small vectors
        else:
            weights: list[float] = [w + adjustment_scalar * x for w, x in zip(perceptron.weights, features)]

        self.perceptron_cache.update_item(cache_id=cache_id, weights=weights, bias=bias)

    def _zscore(self, means: list[float], standard_deviation: list[float], features: list[float]):
        """
        scale the features in the range of `[-3, 3]` with a given means, and standar desviation lists

        args:
            means: list[float] → mean of the dataset columns
            standard_deviation: list[float] → standar desviation of the dataset columns
            features: list[float] → receive a vector of features to scale

        output:
            list[float] → list of features whit the calculated z-score

        time complexity → o(n)

        maths:
            zᵢ = (xᵢ - μ) / σ
        """
        return [(x - means[i]) / standard_deviation[i] for i, x in enumerate(features)]

    def _zscore_dataset(self, dataset: dict[str, any], means: list[float] = None, standard_deviation: list[float] = None):
        """
        normalize the dataset features in the range of `[-3, 3]`

        args:
            features: dict[str, any] → not scalled dataset
            means: list[float] = None → given means
            standard_deviation: list[float] = None → a list of the standar desviation to scale the data whit means and z-score

        return:
            dict[str, any] → normalized features in the range of `-3`, and `3`
            means: list[float] → mean of the dataset columns
            standard_deviation: list[float] → standar desviation of the dataset columns

        time complexity → o(n*f)
        
        maths:
            μ = (Σᵢ₌₁ⁿ xᵢ) / n
            σ = √( Σᵢ₌₁ⁿ (xᵢ - μ)^2 / n )
        """
        # review if is receiving a given means, and standar desviation
        if  means is None and standard_deviation is None:
            num_features: int = len(dataset[0]['features'])
            num_samples: int = len(dataset)

            # calculate the mean for each features (xᵢ) column
            means: list[float] = [
                sum(example['features'][i] for example in dataset) / num_samples 
                for i in range(num_features)
            ]

            # compute the standar desviation for each feature column
            standard_deviation: list[float] = []
            for i in range(num_features):
                column_values: list[float] = [example['features'][i] for example in dataset]

                # determine the variance of the column
                variance: float = sum((x - means[i]) **2 for x in column_values) / (num_samples - 1 if num_samples > 1 else 1)
                std_dev: float = variance ** 0.5

                # avoid division by zero
                if std_dev == 0:
                    std_dev: float = 1

                standard_deviation.append(std_dev)

        # create a new dataset with the normalized features using z-score
        standardized_dataset: list = []
        for example in dataset:
            standardized_example: dict[str, any] = {
                'features': self._zscore(means, standard_deviation, example['features']),
                'label': example['label']
            }

            standardized_dataset.append(standardized_example)

        return standardized_dataset, means, standard_deviation

    def _apply_step_function(self, value: float, epsilon: float = None):
        """
        applies step function to a given value with epsilon

        args:
            value: float → input value to activate
            epsilon: float = None → desviation (grey area) for the step function
        
        output:
            float → 1 if value 0, else 0

        time complexity → o(1)

        maths:
            h(x) = 1 if x ≥ 0, 0 if x < 0
        """
        # if epsilon is not defined apply a normal step function
        if epsilon is None:
            return 1 if value >= 0 else 0
        # validate if value is more than epsilon
        if value > epsilon:
            return 1
        # comparate if value is less than epsilon
        elif value < -epsilon:
            return 0
        # return 0.5 as a grey area if value is in the range of epsilon
        else:
            return 0.5

    def _load_json(self, file_path: str):
        """
        opens a `JSON` file and loads its contents as a `dict`

        args:
            file_path: str → path to the `JSON` dataset file

        output:
            dict → parsed `JSON` data

        time complexity → o(n)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as dataset_file:
                return load(dataset_file)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: `{file_path}`")
        
        except Exception as e:
            raise Exception(f"An Error has been Ocurred: {e}")

if __name__ == "__main__":
    """execute this block only when the script is run directly"""

    # create a cache object
    perceptron_cache = ModelCache(cache_length=10)

    # define the model metadata
    model_metadata = {
        'model_name': "Simple Perceptron", 
        'description': "A simple perceptron trained with the gate `OR`", 
        'author': "Dylan Sutton Chavez"
    }

    # initialize the SimplePerceptron class
    simple_perceptron = SimplePerceptron(perceptron_cache)

    # train the perceptron with specified parameters
    cache_id = simple_perceptron.train(epochs=30, patience=3, labeled_dataset_path='gate-or.json', learning_rate=0.65, model_metadata=model_metadata)

    # load a saved model and make a prediction
    prediction = simple_perceptron.inference(features=[0, 1], cache_id=cache_id)
    print(prediction)

    # define the fine-tuned model metadata
    model_metadata = {
        'model_name': "Simple Perceptron Tuned", 
        'description': "Fine-tuned simple perceptron using the gate `OR`", 
        'author': "Dylan Sutton Chavez"
    }

    # make fine-tuning to the past model
    simple_perceptron.fine_tuning(epochs=10, patience=2, labeled_dataset_path='gate-or.json', learning_rate=0.65, past_model_path='simple-perceptron.2025_10_15.json', model_metadata=model_metadata)

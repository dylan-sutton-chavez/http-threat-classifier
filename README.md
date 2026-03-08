# HTTP Threat Classifier with a Grey Area

A linear classifier extended with a bounded uncertainty region ±ε around the decision boundary. Inputs whose net activation satisfies −ε ≤ z ≤ ε yield a third output state instead of a forced classification, and are escalated to an LLM oracle — a two-tier architecture that trades latency for precision on the inputs where a linear model is least reliable.

---

## The Epsilon Uncertainty Activation Function

Standard step function:

$$
h(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}
$$

Epsilon activation (EUAF):

$$
h(x) = \begin{cases} 1 & x > \varepsilon \\ 0.5 & -\varepsilon \leq x \leq \varepsilon \\ 0 & x < -\varepsilon \end{cases}
$$

The ε parameter controls the width of the uncertainty band. A prediction of `0.5` means the model's net input landed too close to the decision boundary to be trusted. What you do with that signal is up to the caller — escalate to a slower model, log it, or route it differently.

---

## The Salted Polynomial Rolling Hash

Converting variable-length text into a fixed-size numeric vector is a solved problem — but a predictable vectorization is an attack surface. If an adversary can predict where their input lands in vector space, they can craft inputs that evade the classifier.

**epsilon** salts the hash with a 384-bit secret at initialization:

$$
H(S) = \left( \sum_{i=0}^{n-1} U_i \cdot (p + S)^i \right) \bmod m \;\Big/\; S
$$

Where `S` is the salt, `p = 53` is the polynomial base, and `m = 1{,}000{,}000{,}007`. The salt shifts both the base and the output scale, making the resulting vector space private to each deployment.

```python
from secrets import randbits
from text_vectorizer.ngram_hasher import NGramHashVectorizer

salt = randbits(384)
vectorizer = NGramHashVectorizer(salt, chunk_size=150, ngram_size=3)

vectors = vectorizer.vectorized_slices("SELECT * FROM users WHERE id=1 OR 1=1--")
# → list[list[float]]  (one sublist per 150-char chunk)
```

---

## The Learning Rule

Weights and bias update only on misclassified examples:

$$
w_i \leftarrow w_i + \eta \,(y - \hat{y})\, x_i
\qquad
b \leftarrow b + \eta \,(y - \hat{y})
$$

Features are normalized with z-score before training and inference so the learning rate is scale-independent:

$$
z_i = \frac{x_i - \mu_i}{\sigma_i}
$$

The normalization parameters (`μ`, `σ`) are computed from the training set and stored alongside the model weights, so inference uses the same scale as training without requiring the original dataset.

---

## Thread-safe Model Cache

Models live in a shared LRU cache — a `threading.Lock`-guarded `OrderedDict`. Multiple threads can run inference concurrently while a new model is being loaded into a separate slot. When the cache is full, the least-recently-used model is evicted.

```python
from core.perceptron_cache import ModelCache
from core.uncertainty_perceptron import SimplePerceptron

cache = ModelCache(cache_length=10)
perceptron = SimplePerceptron(cache)

cache_id = perceptron.train(
    epochs=30,
    patience=3,
    labeled_dataset_path="data/labeled.json",
    learning_rate=0.65,
    model_metadata={"model_name": "v1", "description": "...", "author": "..."}
)

prediction = perceptron.inference(features=[0.82, 0.44, 0.91], cache_id=cache_id, epsilon=0.12)
# → 0 | 0.5 | 1
```

Dataset format — `list[dict]`:
```json
[
    {"features": [0.82, 0.44, 0.91], "label": 1},
    {"features": [0.11, 0.20, 0.30], "label": 0}
]
```

---

## HTTP Feature Extractors

Four extractors that produce numeric features from the components of an HTTP request:

| Module | Extracts |
|--------|----------|
| `features/uri_syntax.py` | URL length, path depth, query string |
| `features/http_header.py` | Browser type, OS, referer depth, cookie count |
| `features/payload_statistical.py` | Shannon entropy, digit count, special chars, max word length |
| `features/client_profiler.py` | HTTP method encoding |

**Shannon entropy** (normalized):

$$
H_{norm}(X) = \frac{-\sum_{i} p(x_i) \cdot \log_2 p(x_i)}{\log_2 \lvert \Sigma \rvert}
$$

High entropy in a short payload is a strong signal for encoding or obfuscation. Low entropy in a long payload often indicates pattern repetition typical of scanners.

---

## LLM Oracle for the ε-zone

When the perceptron outputs `0.5`, the input can be passed to a language model for a second opinion. The oracle uses structured output (Pydantic) and isolates the untrusted payload inside `<payload>...</payload>` delimiters so prompt injection attempts are classified, not executed.

```python
from feedback.knowledge_client import KnowledgeDistillerLLM

distiller = KnowledgeDistillerLLM("grok-4-fast-reasoning", api_key="xai-...")

result = distiller.inference_query(
    payload="rate: 120 req/min; <payload>SELECT * FROM users WHERE 1=1--</payload>"
)
# → {"label": 1.0, "explanation": "SQL tautology → SQLi. High rate compounds risk."}
```

---

## Running

```bash
# Train and infer on the included OR gate dataset
python -B -m core.uncertainty_perceptron

# LRU cache isolated test
python -B -m core.perceptron_cache

# N-gram vectorizer
python -B -m text_vectorizer.ngram_hasher
```

---

## Dependencies

```
numpy
pydantic
xai-sdk
```

---

## Reference

> Dylan Sutton Chávez (2025).


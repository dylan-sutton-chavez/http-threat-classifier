# Payload Vectorization with O(1) Character N-gram Hashing

> _"A predictable vectorization process creates a critical attack vector: if an adversary could predict where a malicious payload will land in the vector space, they could theoretically "spoof" the isotropic transformation."_ — (Dylan S., 2025)

## 1. Polynomial Rolling Hash & Entropy Salt

The "Polynomial Rolling Hash" is usefull in machine learning because; converts a vecator of "n" dimensions into a unique hash. This proces is verfy fast, and transform long strings into a single vector:

$$
H(S) = \left( \sum_{i=0}^{n-1} s_i \, p^i \right) \bmod m
$$

To prevent model posioning I implemented a salt of 384-bits in this formula, where the resultant hash is divided into the salt (this create small values for the perceptron and make private the hashes), and the polynomial base add the salt to make more unpredecible (an attacker cant predict his vector, he cant manipulate that).

## 2. Slicer Vectorizer Usage

The object `SlicerVectorizer` object have a function named "vectorized_slices"; with a given payload divde into slices, and create n-grams for each slice, then vectorize each ngram with a hashing function and a given salt

### 2.1 Slicer Vectorizer Initialization

The slicer vectorizer object whait to receive the arguments; `salt: int`, `chunk_size: int = 150`, `ngram_size: int = 3`. The arguments refered to sizes are optional.

```python
window_perceptron = SlicerVectorizer(salt) # initialize the slicer vectorizer
```

### 2.2. Transform a String into N-Grams Vector

To use the function `vectorized_slices` you need to pass the next argument; `payload: str`.

```python
payload = "In modern web deve... <script>alert('XSS')</script>... vulnerabilities in database queries if parameterized statements aren’t used."

from secrets import randbits
salt = randbits(384) # generate a random salt (384 bits)

window_perceptron = SlicerVectorizer(salt) # initialize the slicer vectorizer

ngram_vectorized_slices = window_perceptron.vectorized_slices(payload) # vectorize the payload and inject a hashing salt
print(ngram_vectorized_slices)
```

Printed vectors (a list of vectors):

```bash
[[1.4764329611671388e-104,... 4.9323248120543724e-108, 2.7700011189849413e-114]]
```

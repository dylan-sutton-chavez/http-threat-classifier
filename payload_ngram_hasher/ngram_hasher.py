from re import sub, compile

class SlicerVectorizer:
    def __init__(self, hashing_salt: int, chunk_size: int = 150, ngram_size: int = 3):
        """
        initialize the regex compiler pattern and the object
        
        args:
            hashing_salt: int → salt of generation (hash seed)
            chunks: int = 150 → define the size per chunk (ngrams batch)
            ngram_size: int = 3 → define the length for each ngram

        output:
            None

        time complexity → o(1)
        """
        self.allowed_characters = compile(r"[^a-z0-9;'/\\*_\-=+<>\(\)\{\}\[\],.!?:@#$%^&|~`\"]")
        self.salt: int = hashing_salt

        # initialize the chunk and ngram size
        self.chunk_size: int = chunk_size
        self.ngram_size: int = ngram_size

    def vectorized_slices(self, payload: str):
        """
        with a given payload divde int slices, and create ngrams for each slice, then vectorize each ngram with a hashing function and a given salt

        args:
            payload: str → user payload (transmited data)

        return:
            list[list] → list of ngrams batch (vetorized and hashed)

        time complexity → o(n)
        """
        payload_sanitized: str = self.sanitize_payload(payload)

        slices: list[list] = []

        # iterate for each chunk and create ngrams
        for chunk in self._slices_into_parts(payload_sanitized, self.chunk_size):

            ngrams: list[list] = self._slices_into_parts(chunk, self.ngram_size)

            ngram_chunk: list[float] = []

            # convert each ngram into a vectorized hash
            for ngram in ngrams:
                n_gram_vector: list[int] = self._ngram_vector(ngram)
                ngram_vectorized: list[float] = self._ngram_vector_hasher(n_gram_vector)

                ngram_chunk.append(ngram_vectorized)

            slices.append(ngram_chunk)

        return slices
    
    def sanitize_payload(self, payload: str):
        """
        sanitize the payload of the user (lowercase, sanitize characters,...)

        args:
            payload: str → user payload (transmited data)

        output:
            str → user payload normalized

        time complexity → o(n)
        """
        return sub(self.allowed_characters, '', payload.lower())
    
    def _slices_into_parts(self, text: str, size: int):
        """
        slice a text into parts (chunks)

        args:
            text: str → text to slice in part
            size: int → size of the slices

        output:
            list[list] → list with chunks of slices

        time complexity → o(n)
        """
        return [text[i:i+size] for i in range(0, len(text), size)]
    
    def _ngram_vector(self, ngram: str):
        """
        receive a n-gram, and return a vector with the unicode values

        args:
            ngram: str → bag of words whit size n

        output:
            list[int] → list of the ngram characters converted to unicode values

        time complexity → o(n)
        """
        return [ord(c) for c in ngram]
    
    def _ngram_vector_hasher(self, ngram_vector: list[int]):
        """
        hash the ngram vector using the "Polynomial Rolling Hash" and implement a salt to make an impredectible vector

        args:
            ngram_vector: list[int] → ngram vector in unicode format 

        output:
            float → hashed vector with length one

        maths:
            H = ∑ᵢ₌₀ⁿ⁻¹ Uᵢ·(p+S)ⁱ mod m / S
        """
        base_p = 53 # polynomial base for positional sensitive
        modulus_m = 1000000007 # large prime modulus to ensure low collision probability

        # inject the salt to the polynomial base
        effective_base = (base_p + self.salt) % modulus_m

        # ensure the effective base is not trivial (0 or 1)
        if effective_base <= 1:
            effective_base += 2
        
        hash_value = 0 
        p_power = 1

        # iterate for each ngram value and calculare "Polynomial Rolling Hash"
        for ngram_unicode in ngram_vector:
            # Compute the weighted hash term for the current n-gram
            current_term = (ngram_unicode * p_power) % modulus_m
            # Add the current term to the total hash, keeping it within the modulus range
            hash_value = (hash_value + current_term) % modulus_m
            # Increase positional power for the next term
            p_power = (p_power * effective_base) % modulus_m

        return hash_value / self.salt # create an small value whit the secret salt

if __name__ == '__main__':
    from secrets import randbits

    payload = "In modern web development, ensuring security is essential, and developers must always be aware of how users interact with input fields. For instance, a user might type something unusual like <script>alert('XSS')</script> into a comment box, which can help test whether the application properly sanitizes inputs. Similarly, a seemingly harmless username like admin' OR '1'='1 can reveal vulnerabilities in database queries if parameterized statements aren’t used. Even encoded characters such as %3Cimg%20src=x%20onerror=alert(1)%3E illustrate how browsers might interpret unexpected input. Beyond these, careful attention is needed for file uploads, command execution attempts, or URL manipulations like ../etc/passwd, all of which are simulated payloads to help identify weaknesses. By practicing with these examples in a controlled environment, developers can better understand potential threats, strengthen validation routines, and ensure that inputs are properly escaped or sanitized, ultimately creating safer applications."
    salt = randbits(384) # generate a random salt (384 bits)

    window_perceptron = SlicerVectorizer(salt) # initialize the slicer vectorizer

    # sanitize a payload (convert to lower and sanitize some regex characters)
    payload_sanitized: str = window_perceptron.sanitize_payload(payload)
    print(payload_sanitized)
    
    ngram_vectorized_slices: list[list] = window_perceptron.vectorized_slices(payload) # vectorize the payload and inject a hashing salt
    print(ngram_vectorized_slices)
from re import sub, compile

class WindowPerceptron:
    def __init__(self):
        None
        
    # class variable that compile the regex pattern
    _allowed_characters = compile(r"[^a-z0-9;'/\\*_\-=+<>\(\)\{\}\[\],.!?:@#$%^&|~`\"]")
    
    def _slices_into_parts(self, text: str, size: int):
        """
        slice a texxt into parts (chunks)

        args:
            text: str →
            size: int →

        output:
            list[str] →

        time complexity → o(n)
        """
        return [text[i:i+size] for i in range(0, len(text), size)]

    def _sanitize_payload(self, payload: str):
        """
        sanitize the payload of the user (lowercase, sanitize characters,...)

        args:
            payload: str →

        output:
            str →

        time complexity → o(n)
        """
        return sub(self._allowed_characters, '', payload.lower())
    
    def _ngram_vector(self, ngram: str):
        """
        receive a n-gram, and return a vector with the unicode values

        args:
            ngram: str →

        output:
            list[int] → 

        time complexity → o(n)
        """
        return [ord(c) for c in ngram]
    
    def _ngram_vector_hasher(self, ngram_vector: list[int], salt: int):
        """
        hash the ngram vector using the "Polynomial Rolling Hash" and implement a salt to make an impredectible vector

        args:
            ngram_vector: list[int] →
            salt: int →

        output:
            float →

        maths:
            H = ∑ᵢ₌₀ⁿ⁻¹ Uᵢ·(p+S)ⁱ mod m / S
        """
        base_p = 53 # polynomial base for positional sensitive
        modulus_m = 1000000007 # large prime modulus to ensure low collision probability

        # inject the salt to the polynomial base
        effective_base = (base_p + salt) % modulus_m

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

        return hash_value / salt # create an small value whit the secret salt

if __name__ == '__main__':
    from secrets import randbits

    salt = randbits(384)

    payload = "In modern web development, ensuring security is essential, and developers must always be aware of how users interact with input fields. For instance, a user might type something unusual like <script>alert('XSS')</script> into a comment box, which can help test whether the application properly sanitizes inputs. Similarly, a seemingly harmless username like admin' OR '1'='1 can reveal vulnerabilities in database queries if parameterized statements aren’t used. Even encoded characters such as %3Cimg%20src=x%20onerror=alert(1)%3E illustrate how browsers might interpret unexpected input. Beyond these, careful attention is needed for file uploads, command execution attempts, or URL manipulations like ../etc/passwd, all of which are simulated payloads to help identify weaknesses. By practicing with these examples in a controlled environment, developers can better understand potential threats, strengthen validation routines, and ensure that inputs are properly escaped or sanitized, ultimately creating safer applications."

    window_perceptron = WindowPerceptron()

    payload_sanitized = window_perceptron._sanitize_payload(payload)

    for chunk in window_perceptron._slices_into_parts(payload_sanitized, 150):

        ngrams = window_perceptron._slices_into_parts(chunk, 3)

        for ngram in ngrams:
            n_gram_vector = window_perceptron._ngram_vector(ngram)
            print(n_gram_vector)

            print(window_perceptron._ngram_vector_hasher(n_gram_vector, salt))
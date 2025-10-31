from math import log2
from re import split

class PayloadAnalysis:
    def __init__(self, raw_payload: str, payload_window_score: float):
        """
        payload analysis (statics of the payload) and sliding window payload output
        
        args:
            raw_payload: str → payload raw (e.g., pre-vectorized,...)
            payload_window_score: int → output of the simple perceptron payload analysis (sliding window method)

        output:
            None

        time complexity → o(n)
        """
        counted_chars: dict[any, int] = self._count_characters(raw_payload) # count the amount of characters in the payload

        # save metric and calculate the raw payload length 
        self.payload_window_score: float = payload_window_score
        self.payload_length: int = len(raw_payload)

        # analyse and calculate the payload statics
        self.payload_entropy: float = self._shannon_entropy_normalized(counted_chars)
        self.count_digits: int = self._count_digits_in_payload(counted_chars)
        self.cont_special_chars: int = self._count_special_characters(counted_chars)
        self.max_word_length: int = self._max_word_length(raw_payload)

    def _shannon_entropy_normalized(self, counted_chars: dict[any, int]):
        """
        calculate the total shannon entropy normalized from 0 to 1

        args:
            counted_chars: dict[any, int] → characters in the raw payload counted

        output:
            float → shanoon entropy in the payload (scaled from 0 to 1)

        maths:
            H(X) = - Σᵢ₌₁ⁿ p(xᵢ) · log_b(p(xᵢ))
            Hnorm(X) = H(X) / Hmax

        time complexity → o(n)
        """
        # frequency of each character in payload
        frequency: dict[any, float] = {char: counted_chars[char] / self.payload_length for char in counted_chars}

        # shanon entropy for each character
        shannon_entropy: list[float] = [log2(frequency[char]) * frequency[char] for char in frequency]

        # total shanon entropy sum and normalized from 0 to 1 and counter dict
        return -(sum(shannon_entropy)) / log2(len(counted_chars))
    
    def _count_characters(self, raw_payload: str):
        """
        count the frequency of each character in the payload
        
        args:
            raw_payload: str → payload raw (e.g., pre-vectorized,...)

        output:
            dict[any, int] → a dictionary in the format: character and quantity

        time complexity → o(n)
        """
        counter: dict[any, int] = {}

        # count the frequency of each character
        for character in raw_payload:
            # check if the character exists in the conunter
            if character in counter:
                counter[character] += 1
                continue

            counter[character] = 1

        return counter      
    
    def _count_digits_in_payload(self, counted_chars: dict[any, int]):
        """
        count the digits (e.g., 84902.8930 → 9 digits) in the payload

        args:
            counted_chars: dict[any, int] → frequency of each character in the payload

        output:
            int → number of digits in the payload

        time complexity → o(n)
        """
        counter: int = 0

        # itereate for each character in the counted chars
        for char in counted_chars:
            # verify if the chart is a digit
            if char.isdigit():
                counter += counted_chars[char]

        return counter
    
    def _max_word_length(self, payload: str):
        """
        check al the payload and calculate the length of the more length word

        args:
            payload: str → payload raw (e.g., pre-vectorized,...)

        output:
            int → length of the more long word

        time complexity → o(n)
        """
        more_length_word: int = 0

        # iterate in each word of the payload
        for word in split(r'[^a-zA-Z]+', payload):
            word_length = len(word)

            # check if the current word length is biger than the more length word
            if word_length > more_length_word:
                more_length_word = word_length

        return more_length_word
    
    def _count_special_characters(self, counted_chars: dict[any, int]):
        """
        count the special characters in the counted chars dictionary

        args:
            counted_chars: dict[any, int] → frequency of each character in the payload

        output:
            int → total number of the special characters

        time complexity → o(n)
        """
        counter: int = 0

        # iterate in each character of the characters  dict
        for char in counted_chars:
            # check if the current character is in the characters list
            if char in [";", "'", "/", "\\", "*", "_", "-", "=", "+", "<", ">", "(", ")", "{", "}", "[", "]", ",", ".", "!", "?", ":", "@", "#", "$", "%", "^", "&", "|", "~", '"']:
                counter += counted_chars[char]

        return counter
    
if __name__ == '__main__':
    """run this block when the code is runed directly"""

    # create an analysis of the payload and permute the sliding window perceptron output
    payload_analysis = PayloadAnalysis('Hi, I am Dylan. My age its eighteen (18).', 0.5678)

    # structure the object atributes
    payload_analysis_extracted = {
        'payload_window_score': payload_analysis.payload_window_score,
        'payload_entropy': payload_analysis.payload_entropy,
        'payload_length': payload_analysis.payload_length,
        'count_digits': payload_analysis.count_digits,
        'cont_special_chars': payload_analysis.cont_special_chars,
        'max_word_length': payload_analysis.max_word_length
    }

    print(payload_analysis_extracted)
class RabinKarp:

    def __init__(self):
        pass
    """
        This method uses the RabinKarp algorithm to search a given pattern in a given input text.
        @ param pattern - The string pattern that is searched in the text.
        @ param text - The text string in which the pattern is searched.
        @ return a list with the starting indices of pattern occurrences in the text, or None if not found.
        @ raises ValueError if pattern or text is None or empty.
    """


    def search(self, pattern, text):
        if not pattern or not text:
            raise ValueError

        pattern_hash = 0
        text_hash = 0
        result = []

        for i in range(len(pattern)):
            pattern_hash = self.get_rolling_hash_value(pattern[:i+1], None, pattern_hash)
            text_hash = self.get_rolling_hash_value(text[:i+1], None, text_hash)

        for i in range(len(pattern), len(text)):
            if pattern_hash == text_hash and text[i-len(pattern):i] == pattern:
                result.append(i - len(pattern))
            text_hash = self.get_rolling_hash_value(text[i-len(pattern)+1:i+1], text[i-len(pattern)], text_hash)

        if pattern_hash == text_hash and text[-len(pattern):] == pattern:
            result.append(len(text) - len(pattern))

        return result if result else None

    """
         This method calculates the (rolling) hash code for a given character sequence. For the calculation use the 
         base b=29.
         @ param sequence - The char sequence for which the (rolling) hash shall be computed.
         @ param last_character - The character to be removed from the hash when a new character is added.
         @ param previous_hash - The most recent hash value to be reused in the new hash value.
         @ return hash value for the given character sequence using base 29.
    """

    @staticmethod
    def get_rolling_hash_value(sequence, last_character, previous_hash):
        base = 29
        if last_character:
            previous_hash = (previous_hash - ord(last_character) * (base ** (len(sequence) - 1))) * base
        else:
            previous_hash *= base
        return previous_hash + ord(sequence[-1])
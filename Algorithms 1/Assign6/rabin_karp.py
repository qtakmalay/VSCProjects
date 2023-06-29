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
        # TODO
        pass

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
        # TODO
        pass

# sentence/sentence.py

class SentenceBuilder:
    def __init__(self):
        self.current_word = ""
        self.sentence = ""

    def add_letter(self, letter):
        self.current_word += letter

    def add_space(self):
        if self.current_word:
            self.sentence += self.current_word + " "
            self.current_word = ""

    def clear_word(self):
        self.current_word = ""

    def clear_sentence(self):
        self.current_word = ""
        self.sentence = ""

    def get_sentence(self):
        return (self.sentence + self.current_word).strip()

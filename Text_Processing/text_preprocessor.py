# text_preprocessor.py

import re
import spacy
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class TextPreprocessor:
    def __init__(
        self,
        target_column='text',
        include_stopwords=True,
        remove_ats=True,
        word_limit=100,
        tokenizer=None
    ):
        self.target_column = target_column
        self.include_stopwords = include_stopwords
        self.remove_ats = remove_ats
        self.word_limit = word_limit
        self.tokenizer = tokenizer if tokenizer else self.simple_tokenizer

        # Initialize spaCy model for stopwords
        if include_stopwords:
            try:
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                self.stop_words = self.nlp.Defaults.stop_words
            except OSError:
                raise OSError("The spaCy model 'en_core_web_sm' is not installed.")
        else:
            self.stop_words = set()

        # Compile regex patterns
        self.re_pattern = re.compile(r'[^\w\s]')
        self.at_pattern = re.compile(r'@\S+')

    def simple_tokenizer(self, text):
        return text.split()

    def clean_text(self, df):
        if self.target_column not in df.columns:
            raise ValueError(f"The target column '{self.target_column}' does not exist.")

        df = df.copy()
        df[self.target_column] = df[self.target_column].str.lower()

        if self.remove_ats:
            df[self.target_column] = df[self.target_column].str.replace(self.at_pattern, '', regex=True)

        df[self.target_column] = df[self.target_column].str.replace(self.re_pattern, '', regex=True)
        df['tokenized_text'] = df[self.target_column].apply(self.tokenizer)

        if self.include_stopwords:
            df['tokenized_text'] = df['tokenized_text'].apply(
                lambda tokens: [word for word in tokens if word not in self.stop_words and len(word) <= self.word_limit]
            )
        else:
            df['tokenized_text'] = df['tokenized_text'].apply(
                lambda tokens: [word for word in tokens if len(word) <= self.word_limit]
            )

        return df

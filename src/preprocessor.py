import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with necessary tools."""
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Try to load spaCy model, fallback if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            print("spaCy model not found. Using NLTK only.")
            self.use_spacy = False

    def clean_text(self, text):
        """
        Clean text by removing special characters, extra spaces, etc.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove extra punctuation but keep basic ones
        text = re.sub(r"[^\w\s.,!?-]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_text(self, text):
        """
        Tokenize text into words.

        Args:
            text (str): Input text to tokenize

        Returns:
            list: List of tokens
        """
        if self.use_spacy:
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_space]
        else:
            return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """
        Remove stopwords from list of tokens.

        Args:
            tokens (list): List of tokens

        Returns:
            list: List of tokens without stopwords
        """
        return [
            token for token in tokens if token not in self.stop_words and len(token) > 1
        ]

    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their base form.

        Args:
            tokens (list): List of tokens

        Returns:
            list: List of lemmatized tokens
        """
        if self.use_spacy:
            doc = self.nlp(" ".join(tokens))
            return [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct
            ]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text):
        """
        Complete preprocessing pipeline.

        Args:
            text (str): Input text to preprocess

        Returns:
            str: Preprocessed text ready for similarity matching
        """
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize_text(cleaned_text)

        # Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)

        # Join back to string
        processed_text = " ".join(tokens)

        return processed_text

    def preprocess_faq_data(self, faq_list):
        """
        Preprocess a list of FAQ questions.

        Args:
            faq_list (list): List of FAQ dictionaries

        Returns:
            list: List of preprocessed questions
        """
        preprocessed_questions = []

        for faq in faq_list:
            question = faq.get("question", "")
            preprocessed_q = self.preprocess(question)
            preprocessed_questions.append(preprocessed_q)

        return preprocessed_questions

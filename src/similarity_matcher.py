import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class SimilarityMatcher:
    def __init__(self):
        """Initialize the similarity matcher with TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            min_df=1,
            max_df=0.95,
        )
        self.faq_vectors = None
        self.faq_data = None
        self.preprocessed_questions = None

    def fit_faqs(self, faq_data, preprocessed_questions):
        """
        Fit the vectorizer on FAQ data.

        Args:
            faq_data (list): List of FAQ dictionaries
            preprocessed_questions (list): List of preprocessed FAQ questions
        """
        self.faq_data = faq_data
        self.preprocessed_questions = preprocessed_questions

        # Fit and transform FAQ questions
        self.faq_vectors = self.vectorizer.fit_transform(preprocessed_questions)

        print(f"Successfully fitted {len(faq_data)} FAQs")
        print(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def find_best_match(self, user_query, threshold=0.1, top_k=3):
        """
        Find the best matching FAQ for user query.

        Args:
            user_query (str): Preprocessed user query
            threshold (float): Minimum similarity threshold
            top_k (int): Number of top matches to return

        Returns:
            dict: Best match information or None if no good match
        """
        if self.faq_vectors is None:
            raise ValueError("FAQ data not fitted. Call fit_faqs() first.")

        # Transform user query
        user_vector = self.vectorizer.transform([user_query])

        # Calculate similarities
        similarities = cosine_similarity(user_vector, self.faq_vectors).flatten()

        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][
            ::-1
        ]  # Top k in descending order

        matches = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score >= threshold:
                matches.append(
                    {
                        "faq": self.faq_data[idx],
                        "similarity": float(similarity_score),
                        "index": int(idx),
                    }
                )

        if matches:
            return {
                "best_match": matches[0],
                "all_matches": matches,
                "found_match": True,
            }
        else:
            return {
                "best_match": None,
                "all_matches": [],
                "found_match": False,
                "max_similarity": (
                    float(similarities.max()) if len(similarities) > 0 else 0.0
                ),
            }

    def get_similar_questions(self, question_index, top_k=5):
        """
        Get questions similar to a given FAQ question.

        Args:
            question_index (int): Index of the FAQ question
            top_k (int): Number of similar questions to return

        Returns:
            list: List of similar questions with similarity scores
        """
        if self.faq_vectors is None:
            raise ValueError("FAQ data not fitted. Call fit_faqs() first.")

        # Get similarities with all other questions
        question_vector = self.faq_vectors[question_index]
        similarities = cosine_similarity(question_vector, self.faq_vectors).flatten()

        # Get top similar (excluding the question itself)
        similarities[question_index] = -1  # Exclude self
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        similar_questions = []
        for idx in top_indices:
            if similarities[idx] > 0:
                similar_questions.append(
                    {
                        "faq": self.faq_data[idx],
                        "similarity": float(similarities[idx]),
                        "index": int(idx),
                    }
                )

        return similar_questions

    def get_category_questions(self, category, max_questions=5):
        """
        Get questions from a specific category.

        Args:
            category (str): Category to filter by
            max_questions (int): Maximum number of questions to return

        Returns:
            list: List of questions from the category
        """
        if self.faq_data is None:
            return []

        category_questions = [
            faq
            for faq in self.faq_data
            if faq.get("category", "").lower() == category.lower()
        ]

        return category_questions[:max_questions]

    def get_keyword_matches(self, keywords, max_questions=5):
        """
        Get questions that match specific keywords.

        Args:
            keywords (list): List of keywords to match
            max_questions (int): Maximum number of questions to return

        Returns:
            list: List of matching questions
        """
        if self.faq_data is None:
            return []

        keyword_questions = []
        for faq in self.faq_data:
            faq_keywords = faq.get("keywords", [])
            # Check if any keyword matches
            if any(
                kw.lower() in [fk.lower() for fk in faq_keywords] for kw in keywords
            ):
                keyword_questions.append(faq)

        return keyword_questions[:max_questions]

    def get_statistics(self):
        """
        Get statistics about the FAQ matcher.

        Returns:
            dict: Statistics information
        """
        if self.faq_data is None:
            return {}

        categories = [faq.get("category", "unknown") for faq in self.faq_data]
        category_counts = Counter(categories)

        return {
            "total_faqs": len(self.faq_data),
            "vocabulary_size": (
                len(self.vectorizer.vocabulary_) if self.vectorizer else 0
            ),
            "categories": dict(category_counts),
            "most_common_category": (
                category_counts.most_common(1)[0] if category_counts else None
            ),
        }

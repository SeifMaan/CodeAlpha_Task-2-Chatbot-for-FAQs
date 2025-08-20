import json
import random
from datetime import datetime
from preprocessor import TextPreprocessor
from similarity_matcher import SimilarityMatcher


class FAQChatbot:
    def __init__(self, faq_file_path):
        """
        Initialize the FAQ Chatbot.

        Args:
            faq_file_path (str): Path to the FAQ JSON file
        """
        self.preprocessor = TextPreprocessor()
        self.matcher = SimilarityMatcher()
        self.faq_data = []
        self.conversation_history = []
        self.fallback_responses = [
            "I'm sorry, I couldn't find a specific answer to your question. Could you try rephrasing it?",
            "I don't have information about that specific topic. Is there something else I can help you with?",
            "That's a great question! Unfortunately, I don't have that information in my knowledge base.",
            "I'm not sure about that. Could you try asking in a different way or check our menu/contact us directly?",
            "I couldn't find a matching answer. Feel free to contact our restaurant directly for specific inquiries!",
        ]

        # Load FAQ data
        self.load_faq_data(faq_file_path)

        # Fit the similarity matcher
        self._prepare_chatbot()

    def load_faq_data(self, file_path):
        """
        Load FAQ data from JSON file.

        Args:
            file_path (str): Path to FAQ JSON file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.faq_data = data.get("faqs", [])
                self.metadata = data.get("metadata", {})
                print(f"Loaded {len(self.faq_data)} FAQs from {file_path}")
        except FileNotFoundError:
            print(f"FAQ file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            raise

    def _prepare_chatbot(self):
        """Prepare the chatbot by preprocessing FAQ data and fitting the matcher."""
        if not self.faq_data:
            raise ValueError("No FAQ data loaded")

        # Preprocess all FAQ questions
        print("Preprocessing FAQ questions...")
        preprocessed_questions = self.preprocessor.preprocess_faq_data(self.faq_data)

        # Fit the similarity matcher
        print("Training similarity matcher...")
        self.matcher.fit_faqs(self.faq_data, preprocessed_questions)

        print("Chatbot ready!")

    def get_response(self, user_input, confidence_threshold=0.15):
        """
        Get chatbot response for user input.

        Args:
            user_input (str): User's question
            confidence_threshold (float): Minimum confidence for match

        Returns:
            dict: Response with answer, confidence, and metadata
        """
        if not user_input.strip():
            return {
                "answer": "Please ask me a question about our restaurant!",
                "confidence": 0.0,
                "found_match": False,
                "original_question": "",
                "timestamp": datetime.now().isoformat(),
            }

        # Preprocess user input
        processed_input = self.preprocessor.preprocess(user_input)

        # Find best match
        match_result = self.matcher.find_best_match(
            processed_input, threshold=confidence_threshold, top_k=3
        )

        # Prepare response
        if match_result["found_match"]:
            best_match = match_result["best_match"]
            faq = best_match["faq"]

            response = {
                "answer": faq["answer"],
                "confidence": best_match["similarity"],
                "found_match": True,
                "original_question": faq["question"],
                "category": faq.get("category", "general"),
                "faq_id": faq.get("id", "unknown"),
                "alternative_questions": [
                    m["faq"]["question"] for m in match_result["all_matches"][1:3]
                ],
                "timestamp": datetime.now().isoformat(),
            }
        else:
            response = {
                "answer": random.choice(self.fallback_responses),
                "confidence": match_result.get("max_similarity", 0.0),
                "found_match": False,
                "original_question": "",
                "category": "unknown",
                "suggested_questions": self.get_random_questions(3),
                "timestamp": datetime.now().isoformat(),
            }

        # Log conversation
        self.conversation_history.append(
            {
                "user_input": user_input,
                "processed_input": processed_input,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return response

    def get_popular_questions(self, count=6):
        """
        Get popular/recommended questions based on categories.

        Args:
            count (int): Number of questions to return

        Returns:
            list: List of popular questions
        """
        # Define popular categories and their weights
        popular_categories = {
            "hours": 3,
            "menu": 3,
            "delivery": 2,
            "reservations": 2,
            "payment": 1,
            "specials": 2,
        }

        popular_questions = []

        # Get questions from popular categories
        for category, weight in popular_categories.items():
            category_questions = self.matcher.get_category_questions(category, weight)
            popular_questions.extend(category_questions)

        # If we don't have enough, add random ones
        if len(popular_questions) < count:
            remaining = count - len(popular_questions)
            additional = self.get_random_questions(remaining)
            popular_questions.extend(additional)

        # Shuffle and return requested count
        random.shuffle(popular_questions)
        return popular_questions[:count]

    def get_random_questions(self, count=5):
        """
        Get random questions from FAQ data.

        Args:
            count (int): Number of random questions to return

        Returns:
            list: List of random FAQ questions
        """
        if len(self.faq_data) <= count:
            return self.faq_data

        return random.sample(self.faq_data, count)

    def get_questions_by_category(self, category, count=5):
        """
        Get questions from a specific category.

        Args:
            category (str): Category name
            count (int): Number of questions to return

        Returns:
            list: List of questions from the category
        """
        return self.matcher.get_category_questions(category, count)

    def get_all_categories(self):
        """
        Get all available categories.

        Returns:
            list: List of unique categories
        """
        categories = set()
        for faq in self.faq_data:
            categories.add(faq.get("category", "general"))
        return sorted(list(categories))

    def search_questions(self, keywords, count=5):
        """
        Search questions by keywords.

        Args:
            keywords (list): List of keywords
            count (int): Maximum number of results

        Returns:
            list: List of matching questions
        """
        return self.matcher.get_keyword_matches(keywords, count)

    def get_statistics(self):
        """
        Get chatbot statistics.

        Returns:
            dict: Statistics about the chatbot
        """
        stats = self.matcher.get_statistics()
        stats.update(
            {
                "conversations": len(self.conversation_history),
                "domain": self.metadata.get("domain", "Unknown"),
                "last_updated": self.metadata.get("last_updated", "Unknown"),
            }
        )
        return stats

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def export_conversation(self, file_path):
        """
        Export conversation history to JSON file.

        Args:
            file_path (str): Path to save conversation history
        """
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(self.conversation_history, file, indent=2, ensure_ascii=False)
            print(f"Conversation exported to {file_path}")
        except Exception as e:
            print(f"Error exporting conversation: {e}")


# Helper function to create FAQ JSON file
def create_sample_faq_file(file_path):
    """Create a sample FAQ JSON file for testing."""
    sample_data = {
        "faqs": [
            {
                "id": 1,
                "question": "What are your restaurant hours?",
                "answer": "We're open Monday through Thursday from 11 AM to 10 PM, Friday and Saturday from 11 AM to 11 PM, and Sunday from 12 PM to 9 PM.",
                "category": "hours",
                "keywords": ["hours", "open", "close", "time"],
            },
            {
                "id": 2,
                "question": "Do you offer delivery service?",
                "answer": "Yes! We offer delivery within a 5-mile radius for orders over $20. Delivery usually takes 30-45 minutes.",
                "category": "delivery",
                "keywords": ["delivery", "deliver", "order"],
            },
        ],
        "metadata": {
            "domain": "Restaurant",
            "total_faqs": 2,
            "last_updated": "2025-01-20",
        },
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(sample_data, file, indent=2, ensure_ascii=False)

    print(f"Sample FAQ file created: {file_path}")

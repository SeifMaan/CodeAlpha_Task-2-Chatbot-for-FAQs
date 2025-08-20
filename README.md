# 🍽️ Restaurant FAQ Chatbot

An intelligent, AI-powered chatbot designed specifically for restaurants to handle frequently asked questions. Built with Python and Streamlit, this chatbot uses natural language processing to understand customer queries and provide accurate, helpful responses.

## ✨ Features

- **🤖 Intelligent FAQ Matching**: Uses TF-IDF and cosine similarity to find the best answers to customer questions
- **📱 User-Friendly Web Interface**: Clean, responsive Streamlit web app with example questions
- **🗂️ Category-Based Organization**: FAQs organized into logical categories (hours, menu, delivery, etc.)
- **📊 Conversation Analytics**: Track chatbot performance and user interactions
- **🎯 Contextual Understanding**: Handles variations in how questions are asked
- **🔊 Multi-language Support**: Ready for internationalization (though currently focused on English)
- **📝 Export Capabilities**: Save conversation history for analysis

## 🛠️ Tech Stack

- **Python 3.8+** - Core programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning for text processing
- **NLTK** - Natural language processing toolkit
- **spaCy** - Advanced NLP capabilities
- **TF-IDF Vectorization** - Text similarity matching

## 📋 Prerequisites

Before running this application, ensure you have:

- Python 3.8 or higher installed
- pip (Python package manager)
- Internet connection (for initial package downloads)

## 🚀 Installation & Setup

1. **Clone or download the project files**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate    # On Windows
   ```
3. **Install required dependencies**
   pip install -r requirements.txt

## 📁 Project Structure

restaurant-faq-chatbot/
├── src/
│ ├── chatbot.py # Main chatbot class
│ ├── text_preprocessor.py # Text cleaning and preprocessing
│ ├── similarity_matcher.py # FAQ matching logic
│ └── app.py # Streamlit web application
├── data/
│ └── faqs.json # Restaurant FAQ database
├── requirements.txt # Python dependencies
└── README.md # Project documentation

# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging

# # Initialize Flask app
# app = Flask(__name__)

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger()

# # Load the expanded dataset
# faq_data = pd.read_csv("expanded_medical_faqs.csv")

# # Load SpaCy's pre-trained NER model
# nlp = spacy.load("en_core_web_sm")

# # Vectorize the Queries for similarity matching
# vectorizer = TfidfVectorizer()
# query_vectors = vectorizer.fit_transform(faq_data["Query"])

# def get_response(user_query):
#     """
#     Generate a response to the user query using similarity matching
#     and entity recognition.
#     """
#     logger.info("Processing user query: %s", user_query)

#     # Perform NER on the user query
#     doc = nlp(user_query)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]

#     # Compute similarity between user query and FAQ queries
#     user_vector = vectorizer.transform([user_query])
#     similarities = cosine_similarity(user_vector, query_vectors)
#     best_match_idx = similarities.argmax()

#     # Get the most relevant response
#     response = faq_data.iloc[best_match_idx]["Response"]

#     # Add entity recognition info to the response
#     if entities:
#         response += "\n\nDetected entities: " + ", ".join([f"{text} ({label})" for text, label in entities])

#     logger.info("Generated response: %s", response)
#     return response

# @app.errorhandler(500)
# def handle_internal_error(error):
#     """
#     Handle internal server errors gracefully.
#     """
#     logger.error("Server error: %s", error)
#     return jsonify({"error": "An internal error occurred. Please try again later."}), 500

# @app.errorhandler(404)
# def handle_not_found_error(error):
#     """
#     Handle 404 errors gracefully.
#     """
#     logger.error("Page not found: %s", error)
#     return jsonify({"error": "Endpoint not found."}), 404

# # Define a route for the chatbot API
# @app.route("/chat", methods=["POST"])
# def chat():
#     user_query = request.json.get("query")
#     if not user_query:
#         return jsonify({"error": "No query provided."}), 400

#     response = get_response(user_query)
#     return jsonify({"response": response})

# # Define a route for the frontend
# @app.route("/")
# def index():
#     return render_template("index.html")

# if __name__ == "__main__":
#     logger.info("Starting Medical Chatbot application.")
#     app.run(debug=True)



from flask import Flask, request, jsonify, render_template
import pandas as pd
import logging
import signal
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load the expanded dataset
faq_data = pd.read_csv("expanded_medical_faqs.csv")

# Prepare data for deep learning
questions = faq_data['Query'].tolist()
answers = faq_data['Response'].tolist()

# Tokenizer and padding for deep learning
max_words = 10000
max_len = 20

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(questions)
question_sequences = tokenizer.texts_to_sequences(questions)
question_padded = pad_sequences(question_sequences, maxlen=max_len, padding='post')

# Prepare answers as categorical data
answer_tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
answer_tokenizer.fit_on_texts(answers)
answer_sequences = answer_tokenizer.texts_to_sequences(answers)
answer_padded = pad_sequences(answer_sequences, maxlen=max_len, padding='post')

# Create the deep learning model
embedding_dim = 64
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(len(answer_tokenizer.word_index) + 1, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
logger.info("Model compiled successfully.")

# Train the model (placeholder, replace with actual training logic)
# model.fit(question_padded, answer_padded, epochs=10, batch_size=32)
logger.info("Model training skipped for this example.")

# TF-IDF Vectorizer for traditional similarity matching
tfidf_vectorizer = TfidfVectorizer()
faq_tfidf_matrix = tfidf_vectorizer.fit_transform(faq_data['Query'])

# Function to find the most similar question using TF-IDF
def find_similar_question(user_input):
    try:
        user_tfidf = tfidf_vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_tfidf, faq_tfidf_matrix)
        best_match_idx = similarity_scores.argmax()
        best_score = similarity_scores[0, best_match_idx]
        
        if best_score > 0.2:  # Threshold for similarity
            return faq_data.iloc[best_match_idx]['Response']
        else:
            return "I'm sorry, I couldn't find an answer to your question. Could you rephrase it?"
    except Exception as e:
        logger.error(f"Error finding similar question: {e}")
        return "An error occurred while processing your question. Please try again."

# Function to get an answer using the trained model
def get_answer_from_model(user_input):
    try:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        prediction = model.predict(padded_sequence)
        answer_index = prediction.argmax(axis=-1)[0]
        reverse_answer_map = {v: k for k, v in answer_tokenizer.word_index.items()}
        answer_tokens = [reverse_answer_map.get(idx, '') for idx in range(1, answer_index + 1)]
        return ' '.join(answer_tokens) or "I'm sorry, I couldn't process your question."
    except Exception as e:
        logger.error(f"Error in model prediction: {e}")
        return "An error occurred while processing your question. Please try again."

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# API route for chatbot
@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        user_input = request.form['user_input']
        if user_input.strip().lower() == "exit":
            return jsonify({"response": "Goodbye! The chatbot is now shutting down."})

        # Use both methods and combine responses
        tfidf_response = find_similar_question(user_input)
        dl_response = get_answer_from_model(user_input)

        response = f"{tfidf_response}\n"
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in /get_answer: {e}")
        return jsonify({"response": "An error occurred. Please try again later."})

# Signal handler for graceful shutdown
def handle_exit_signal(signal_received, frame):
    print("\nGoodbye! Shutting down the chatbot.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit_signal)

# Run the app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Error starting the app: {e}")


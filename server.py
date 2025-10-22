from flask import Flask, request, jsonify
from flask_cors import CORS
import chatbot  # make sure your chatbot.py is in the same directory and has the function get_chatbot_response

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    response = chatbot.get_chatbot_response(user_message)
    return jsonify({'response': response})

@app.route('/')
def home():
    return "EduCRM Chatbot Server is running. Use POST /chat endpoint to interact."

if __name__ == '__main__':
    app.run(debug=True, port=8000)

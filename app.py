import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load the trained model
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('email', '')
    
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400
    
    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    
    spam_prob = probabilities[1] if prediction == 'spam' else probabilities[0]
    
    return jsonify({
        'prediction': prediction,
        'confidence': float(spam_prob) * 100
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

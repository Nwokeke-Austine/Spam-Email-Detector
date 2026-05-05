# Spam Email Detector

A web-based spam email detection system using Naive Bayes classification.

## Files Included
- `index.html` - Frontend web interface (black & white theme)
- `app.py` - Flask backend API server
- `train_model.py` - Model training script
- `spam_model.pkl` - Trained Naive Bayes model

## Setup Instructions

1. Install required packages:
   ```bash
   pip install flask flask-cors scikit-learn pandas numpy
   ```

2. Train the model (if not already trained):
   ```bash
   python train_model.py
   ```

3. Start the Flask backend:
   ```bash
   python app.py
   ```
   Server will run on http://localhost:5000

4. Open `index.html` in your web browser
   - You can double-click the file or
   - Use a local server: `python -m http.server 8000`
   - Then visit http://localhost:8000

## Usage

1. Enter or paste email text into the text area
2. Click "Analyze Email" to check if it's spam
3. View the result with confidence percentage
4. Try the example emails to see how it works

## Technology Stack
- Backend: Python, Flask, scikit-learn
- Frontend: HTML, CSS, JavaScript
- ML Algorithm: Naive Bayes (MultinomialNB)
- Classification: Supervised learning

## Color Theme
Black and white minimalist design with gradient accents

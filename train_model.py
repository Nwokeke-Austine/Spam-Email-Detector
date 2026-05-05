import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Training data - spam and ham emails
spam_emails = [
    "FREE GIFT! Click here to claim your prize now!!!",
    "Congratulations! You've won $1,000,000! Reply with your bank details",
    "URGENT: Your account will be closed. Click here immediately",
    "Get rich quick! Make money from home working just 2 hours daily",
    "Limited time offer! Buy now and get 90% discount!!!",
    "You have been selected for a special offer. Act now!",
    "WINNER! Claim your free iPhone now by clicking this link",
    "Lowest prices on medications. No prescription needed!",
    "Increase your income by 500% working from home",
    "Meet hot singles in your area tonight! Click here",
    "Your PayPal account has been compromised. Verify now",
    "Free casino credits! Play now and win big!!!",
    "Lose weight fast with this one weird trick",
    "CONGRATULATIONS!!! You are our lucky winner today",
    "Get free money transferred to your account today",
]

ham_emails = [
    "Hi, can we schedule a meeting for next Tuesday?",
    "Here are the documents you requested yesterday",
    "Thanks for your help with the project. Much appreciated!",
    "The team meeting has been moved to 3 PM tomorrow",
    "Please review the attached report and let me know your thoughts",
    "Reminder: Project deadline is this Friday",
    "Great work on the presentation today!",
    "Can you send me the updated spreadsheet?",
    "Looking forward to our call this afternoon",
    "The quarterly results look promising this year",
    "Happy birthday! Hope you have a wonderful day",
    "Just checking in to see how things are going",
    "The package will be delivered tomorrow morning",
    "Let me know if you need any assistance",
    "Thanks for attending the conference call",
]

# Create training data
X_train = spam_emails + ham_emails
y_train = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)

# Create and train the pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)

# Save the model
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")

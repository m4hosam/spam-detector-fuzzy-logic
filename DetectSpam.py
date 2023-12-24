
import math
from transformers import pipeline
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from SpamFuzzyController import SpamFuzzyController
from SpamWords import SpamWords
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

spam_filter = pipeline("text-classification", model="NotShrirang/albert-spam-filter")

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_subject_spam_score(subject, spam_words):
    # Tokenize the subject
    tokens = word_tokenize(subject.lower())

    # Remove stop words
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]

    # Calculate the total score based on the presence of spam words
    total_score = sum(1 for token in tokens if token in spam_words)

    # Normalize the score using the sigmoid function
    normalized_score = sigmoid(total_score - len(spam_words)/2)  # Adjusted the normalization
    rounded_score = round(normalized_score*100, 3)

    return rounded_score

def calculate_body_spam_score(email_body):
    # Use ALBERT model to get spam probability
    albert_output = spam_filter(email_body, top_k=2)

    print(f"albert_output: {albert_output}")

    if albert_output[0]['label'] == 'Spam':
      albert_spam_prob = albert_output[0]['score']
      albert_spam_prob = round(albert_spam_prob*100, 3)
    else:
      albert_spam_prob = albert_output[1]['score']
      albert_spam_prob = round(albert_spam_prob*100, 3)


    return albert_spam_prob



if __name__ == '__main__':
    
    # Test the spam detector
    email_subject = "Invitation to Exclusive Webinar"
    email_body = "Hello, Thank you for your interest in our product. We are happy to offer you a 20% discount on your next purchase. Please click on the link below to claim your discount."
    # Calculate spam term frequency score for email subject
    spam_words = {'claim', 'free', 'win', 'offer', 'click', 'money'}

    subject_spam_score = calculate_subject_spam_score(email_subject, spam_words)
    print(f"subject_spam_score: {subject_spam_score}")
    albert_output = calculate_body_spam_score(email_body)
    print(f"albert_output: {albert_output}")

    spamfuz = SpamFuzzyController()
    spamfuz.fuzzy_initialize()
    spam_score_fuzzy = spamfuz.fuzzy_predict(albert_output, subject_spam_score)

    exit(0)
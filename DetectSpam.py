
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


   

    # print(f"subject_spam_score: {subject_spam_score}")
    # print(f"albert_output: {albert_output}")
    # print(f"albert-spam-filter score: {albert_spam_prob}")

    # # Set input values for the fuzzy logic system
    # spam_detector.input['albert_spam_probability'] = albert_spam_prob
    # spam_detector.input['subject_spam_score'] = subject_spam_score

    #check if fuzzyfication is working properly
    #print(spam_detector.input['albert_spam_probability'])
    #print(spam_detector.input['subject_spam_score'])

    #check which rules are being activated
    #print(spam_detector.rules)

    # Compute the result
    # spam_detector.compute()

    # Get the final spam probability
    # final_spam_prob = spam_detector.output['final_spam_probability']

    return albert_spam_prob



if __name__ == '__main__':
    
    # Test the spam detector
    email_subject = "Claim 100% more free money now!"
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

    #email_body = "Subject: jennifer sends them to their final destination . designated as a private key 4 . validate public keys . someone wants to meet you ! who your match could be , find here to the purported owner . you"
    # spam_probability = test_spam_detector(email_subject, email_body)
    # spam_probability = round(spam_probability, 3)
    # print(f"Spam Probability: {spam_probability}%")
    exit(0)
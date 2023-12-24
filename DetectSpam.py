import math
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from SpamFuzzyController import SpamFuzzyController
from SpamWords import SpamWords
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_subject_spam_score(subject):
    # Convert subject to lowercase for case-insensitive matching
    email_subject_lower = subject.lower().strip()

    # Initialize an empty list to store the spam words present in the email subject
    occurrence_words = ""

    # Adding the spam words to the list if they are present in the email subject
    for word in SpamWords:
        if word.lower() in email_subject_lower:
            occurrence_words += word.lower() + " "

    occurrence_words = occurrence_words.strip()
    # print(f"occurrence_words: {occurrence_words}")

    # devide the characters of the occurence words by the total number of characters in the email subject
    normalized_score = len(occurrence_words) / len(email_subject_lower)
    # Calculate the spam term frequency score for email subject
    rounded_score = round(normalized_score*100, 3)

    return rounded_score

def calculate_body_spam_score(email_body):
    from transformers import pipeline
    # Use ALBERT model to get spam probability
    spam_filter = pipeline("text-classification", model="NotShrirang/albert-spam-filter")
    albert_output = spam_filter(email_body, top_k=2)

    print(f"albert_output: {albert_output}")

    if albert_output[0]['label'] == 'Spam':
      albert_spam_prob = albert_output[0]['score']
    else:
      albert_spam_prob = albert_output[1]['score']

    albert_spam_prob = round(albert_spam_prob*100, 3)
    return albert_spam_prob


if __name__ == '__main__':
    
    # Test the spam detector
    email_subject = "Claim Free money "
    email_body = "Hello, Thank you for your invite. See you tomorrow."
    # Calculate spam term frequency score for email subject

    subject_spam_score = calculate_subject_spam_score(email_subject)
    print(f"subject_spam_score: {subject_spam_score}")
    albert_output = calculate_body_spam_score(email_body)
    print(f"albert_output: {albert_output}")

    spamfuz = SpamFuzzyController()
    spamfuz.fuzzy_initialize()
    spam_score_fuzzy = spamfuz.fuzzy_predict(albert_output, subject_spam_score)

    exit(0)
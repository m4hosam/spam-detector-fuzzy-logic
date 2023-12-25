from transformers import pipeline
from SpamFuzzyController import SpamFuzzyController
from SpamWords import SpamWords
from emails import email_examples
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)


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
    # Use ALBERT model to get spam probability
    spam_filter = pipeline("text-classification", model="NotShrirang/albert-spam-filter")
    # spam_filter = pipeline("text-classification", model="mshenoda/roberta-spam")
    albert_output = spam_filter(email_body, top_k=2)

    # print(f"albert_output: {albert_output}")

    # if albert_output[0]['label'] == 'LABEL_1':
    if albert_output[0]['label'] == 'Spam':
      albert_spam_prob = albert_output[0]['score']
    else:
      albert_spam_prob = albert_output[1]['score']

    albert_spam_prob = round(albert_spam_prob*100, 3)
    return albert_spam_prob


if __name__ == '__main__':
    
    # Test the spam detector
    # print(email_examples[0]['subject'])
    # print(email_examples[0]['body'])
    # Testing
    for email in email_examples:
        # print(f"subject: {email['subject']}")
        # print(f"body: {email['body']}")
        print(f"spam: {email['spam']}")
        subject_spam_score = calculate_subject_spam_score(email['subject'])
        albert_output = calculate_body_spam_score(email['body'])
        spamfuz = SpamFuzzyController()
        spamfuz.fuzzy_initialize()
        spam_score_fuzzy = spamfuz.fuzzy_predict(albert_output, subject_spam_score)
        print(f"subject spam score: {subject_spam_score}", "% spam")
        print(f"spam model output: {albert_output}", "% spam")
        print("Spam Score Fuzzy: ", spam_score_fuzzy, "% spam")
        print("-------------------------------------------------")


    # Calculate spam term frequency score for email subject
    # subject_spam_score = calculate_subject_spam_score(email_examples[8]['subject'])
    # albert_output = calculate_body_spam_score(email_examples[8]['body'])

    # spamfuz = SpamFuzzyController()
    # spamfuz.fuzzy_initialize()
    # spam_score_fuzzy = spamfuz.fuzzy_predict(albert_output, subject_spam_score)

    # print(f"subject spam score: {subject_spam_score}", "% spam")
    # print(f"spam model output: {albert_output}", "% spam")
    # print("Spam Score Fuzzy: ", spam_score_fuzzy, "% spam")

    exit(0)
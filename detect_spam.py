import numpy as np
import skfuzzy as fuzz

# Linguistic variables: Message length and Frequency of suspicious words
message_length = np.arange(0, 101, 1)  # Assuming message length in characters
frequency = np.arange(0, 11, 1)  # Frequency of suspicious words

# Generate fuzzy membership functions for message length
short = fuzz.trimf(message_length, [0, 25, 50])
medium = fuzz.trimf(message_length, [25, 50, 75])
long = fuzz.trimf(message_length, [50, 75, 100])

# Generate fuzzy membership functions for frequency of suspicious words
suspicious_low = fuzz.trimf(frequency, [0, 3, 5])
suspicious_medium = fuzz.trimf(frequency, [4, 6, 8])
suspicious_high = fuzz.trimf(frequency, [7, 10, 10])

# Function to evaluate spamminess based on input message length and frequency of suspicious words
def detect_spam(message_length_input, frequency_input):
    # Fuzzify the inputs
    membership_short = fuzz.interp_membership(message_length, short, message_length_input)
    membership_medium = fuzz.interp_membership(message_length, medium, message_length_input)
    membership_long = fuzz.interp_membership(message_length, long, message_length_input)
    
    membership_low = fuzz.interp_membership(frequency, suspicious_low, frequency_input)
    membership_medium_freq = fuzz.interp_membership(frequency, suspicious_medium, frequency_input)
    membership_high = fuzz.interp_membership(frequency, suspicious_high, frequency_input)
    
    # Rule base
    # Rule 1: IF message length is short AND frequency of suspicious words is high THEN classify as spam with high confidence.
    rule1 = np.fmin(membership_short, membership_high)
    
    # Rule 2: IF message length is medium AND frequency of suspicious characters is moderate THEN classify as potential spam.
    rule2 = np.fmin(membership_medium, membership_medium_freq)
    
    # Rule 3: IF message length is long AND frequency of suspicious words is low THEN classify as not spam.
    rule3 = np.fmin(membership_long, membership_low)
    
    # Combine rules
    aggregated = np.fmax(rule1, np.fmax(rule2, rule3))
    
    # Defuzzification (centroid method)
    spam_score = fuzz.defuzz(frequency, aggregated, 'centroid')
    return spam_score

# Test message details (for demonstration)
test_message_length = 30  # Short message
test_frequency = 8  # High frequency of suspicious words

# Detect spamminess of the test message
spam_score = detect_spam(test_message_length, test_frequency)

# Threshold to determine if it's spam or not (adjust as needed)
spam_threshold = 5

# Decide if it's spam or not based on the spam score
if spam_score >= spam_threshold:
    print("This message is classified as spam.")
else:
    print("This message is not classified as spam.")

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz


def message_spam_mf():
    # Sample linguistic variable: Frequency of suspicious words
    # Membership function definition
    frequency = np.arange(0, 101, 1)

    # Generate fuzzy membership functions
    not_spam = fuzz.trapmf(frequency, [0,0, 20, 40])
    maybe_spam = fuzz.trapmf(frequency, [20,40, 60, 80])
    definite_spam = fuzz.trapmf(frequency, [60,80, 100, 100])

    # Visualize the membership functions

    plt.figure()
    plt.plot(frequency, not_spam, 'b', label='Not Spam')
    plt.plot(frequency, maybe_spam, 'g', label='Maybe Spam')
    plt.plot(frequency, definite_spam, 'r', label='Spam')
    plt.title('Membership Functions for Message Spam probability')
    plt.legend()
    plt.show()

def subject_spam_mf():
    # Sample linguistic variable: Frequency of suspicious words
    # Membership function definition
    frequency = np.arange(0, 101, 1)

    # Generate fuzzy membership functions
    not_spam = fuzz.trapmf(frequency, [0,0, 20, 40])
    maybe_spam = fuzz.trapmf(frequency, [20,40, 60, 80])
    definite_spam = fuzz.trapmf(frequency, [60,80, 100, 100])

    # Visualize the membership functions

    plt.figure()
    plt.plot(frequency, not_spam, 'b', label='Not Spam')
    plt.plot(frequency, maybe_spam, 'g', label='Maybe Spam')
    plt.plot(frequency, definite_spam, 'r', label='Spam')
    plt.title('Membership Functions for Message Spam probability')
    plt.legend()
    plt.show()


message_spam_mf()
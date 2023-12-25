import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

def plot_membership_functions(ax, frequency, not_spam, maybe_spam, definite_spam, title):
    ax.plot(frequency, not_spam, 'b', label='Not Spam')
    ax.plot(frequency, maybe_spam, 'g', label='Maybe Spam')
    ax.plot(frequency, definite_spam, 'r', label='Spam')
    ax.set_title(title)
    ax.legend()

def plot_all_membership_functions():
    # Sample linguistic variable: Frequency of suspicious words
    # Membership function definition
    frequency = np.arange(0, 101, 1)

    # Generate fuzzy membership functions for message spam
    not_spam_message = fuzz.trapmf(frequency, [0, 0, 20, 40])
    maybe_spam_message = fuzz.trapmf(frequency, [20, 40, 60, 80])
    definite_spam_message = fuzz.trapmf(frequency, [60, 80, 100, 100])

    # Generate fuzzy membership functions for subject spam
    not_spam_subject = fuzz.trimf(frequency, [0, 0, 51])
    maybe_spam_subject = fuzz.trimf(frequency, [0, 51, 101])
    definite_spam_subject = fuzz.trimf(frequency, [51, 101, 101])

    # Output membership function using Gaussian centered around 50
    output_not_spam = fuzz.gaussmf(frequency, 0, 20)
    output_maybe_spam = fuzz.gaussmf(frequency, 50, 20)
    output_definite_spam = fuzz.gaussmf(frequency, 100, 20)

    # Plot all membership functions on the same figure using subplots
    _, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot membership functions for message spam
    plot_membership_functions(axs[0], frequency, not_spam_message, maybe_spam_message, definite_spam_message, 'Message Spam Probability')

    # Plot membership functions for subject spam
    # plot_membership_functions(axs[1], frequency, not_spam_subject, maybe_spam_subject, definite_spam_subject, 'Subject Spam Probability')

    # Plot membership functions for output
    plot_membership_functions(axs[1], frequency, output_not_spam, output_maybe_spam, output_definite_spam, 'Output Membership Function')

    plt.tight_layout()
    plt.show()

plot_all_membership_functions()

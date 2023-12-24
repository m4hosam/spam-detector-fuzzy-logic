import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class SpamFuzzyController:
    _spam_percentage = None

    def __init__(self):
        print('Initializing Fuzzy Logic')

    def fuzzy_initialize(self):
        # Creating two fuzzy input variables and one output fuzzy variable
        albert_spam_probability = ctrl.Antecedent(np.arange(0, 101, 1), 'albert_spam_probability')
        tfidf_score = ctrl.Antecedent(np.arange(0, 101, 1), 'tfidf_score')
        predict_spam = ctrl.Consequent(np.arange(0, 100, 1), 'Spam_Prediction')

        # Creating memberships for tweet model - Input Variable
        albert_spam_probability['not_spam'] = fuzz.trapmf(albert_spam_probability.universe, [0,0, 20, 40])
        albert_spam_probability['maybe_spam'] = fuzz.trapmf(albert_spam_probability.universe, [20,40, 60, 80])
        albert_spam_probability['definite_spam'] = fuzz.trapmf(albert_spam_probability.universe, [60,80, 100, 100])
        # albert_spam_probability['not_spam'] = fuzz.trimf(albert_spam_probability.universe, [0, 0, 49])
        # albert_spam_probability['maybe_spam'] = fuzz.trimf(albert_spam_probability.universe, [50, 50, 101])
        # albert_spam_probability['definite_spam'] = fuzz.trimf(albert_spam_probability.universe, [50, 101, 101])
        
        # Creating memberships for user model - Input Variable
        # tfidf_score['not_spam'] = fuzz.trapmf(tfidf_score.universe, [0,0, 20, 40])
        # tfidf_score['maybe_spam'] = fuzz.trapmf(tfidf_score.universe, [20,40, 60, 80])
        # tfidf_score['definite_spam'] = fuzz.trapmf(tfidf_score.universe, [60,80, 100, 100])
        tfidf_score['not_spam'] = fuzz.trimf(tfidf_score.universe, [0, 0, 51])
        tfidf_score['maybe_spam'] = fuzz.trimf(tfidf_score.universe, [0, 51, 101])
        tfidf_score['definite_spam'] = fuzz.trimf(tfidf_score.universe, [51, 101, 101])

        # Creating memberships for tweet model - Output Variable
        predict_spam['not_spam'] = fuzz.trapmf(predict_spam.universe, [0,0, 20, 40])
        predict_spam['maybe_spam'] = fuzz.trapmf(predict_spam.universe, [20,40, 60, 80])
        predict_spam['spam'] = fuzz.trapmf(predict_spam.universe, [60,80, 100, 100])
        # predict_spam['not_spam'] = fuzz.trimf(predict_spam.universe, [0, 0, 51])
        # predict_spam['maybe_spam'] = fuzz.trimf(predict_spam.universe, [0, 51, 101])
        # predict_spam['spam'] = fuzz.trimf(predict_spam.universe, [51, 101, 101])

        # # view a graph showing the memberships of the variables initialized
        # albert_spam_probability.view()
        # tfidf_score.view()
        # predict_spam.view()

        # initiating rules for the spam fuzzy controller 3^2 probability therefore nine rules will be applied
        rule1 = ctrl.Rule(albert_spam_probability['not_spam'] & tfidf_score['not_spam'], predict_spam['not_spam'])
        rule2 = ctrl.Rule(albert_spam_probability['not_spam'] & tfidf_score['maybe_spam'], predict_spam['not_spam'])
        rule3 = ctrl.Rule(albert_spam_probability['not_spam'] & tfidf_score['definite_spam'], predict_spam['maybe_spam'])

        rule4 = ctrl.Rule(albert_spam_probability['maybe_spam'] & tfidf_score['not_spam'], predict_spam['not_spam'])
        rule5 = ctrl.Rule(albert_spam_probability['maybe_spam'] & tfidf_score['maybe_spam'], predict_spam['spam'])
        rule6 = ctrl.Rule(albert_spam_probability['maybe_spam'] & tfidf_score['definite_spam'], predict_spam['spam'])

        rule7 = ctrl.Rule(albert_spam_probability['definite_spam'] & tfidf_score['not_spam'], predict_spam['maybe_spam'])
        rule8 = ctrl.Rule(albert_spam_probability['definite_spam'] & tfidf_score['maybe_spam'], predict_spam['spam'])
        rule9 = ctrl.Rule(albert_spam_probability['definite_spam'] & tfidf_score['definite_spam'], predict_spam['spam'])

        # Add the rules to a new ControlSystem
        predict_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self._spam_percentage = ctrl.ControlSystemSimulation(predict_control)

    def fuzzy_predict(self, albert_spam_probability, tfidf_score_proba):
        spam_percentage = self._spam_percentage
        spam_percentage.input['albert_spam_probability'] = albert_spam_probability
        spam_percentage.input['tfidf_score'] = tfidf_score_proba

        # Crunch the numbers
        spam_percentage.compute()

        print(" albert_spam_probability - {0} | tfidf_score = {1} | Fuzzy Prediction = {2}".format(albert_spam_probability, tfidf_score_proba,
                                                                                      spam_percentage.output[
                                                                                          'Spam_Prediction']))
        final_spam_score = spam_percentage.output['Spam_Prediction']
        return int(round(final_spam_score))


if __name__ == '__main__':
    spamfuz = SpamFuzzyController()
    spamfuz.fuzzy_initialize()
    spam_score_fuzzy = spamfuz.fuzzy_predict(40, 40)
    spam_score_fuzzy = spamfuz.fuzzy_predict(50, 50)
    spam_score_fuzzy = spamfuz.fuzzy_predict(10, 10)
    spam_score_fuzzy = spamfuz.fuzzy_predict(40, 60)
    spam_score_fuzzy = spamfuz.fuzzy_predict(60, 60)
    spam_score_fuzzy = spamfuz.fuzzy_predict(80, 80)
    spam_score_fuzzy = spamfuz.fuzzy_predict(90, 90)
    spam_score_fuzzy = spamfuz.fuzzy_predict(20, 90)
    spam_score_fuzzy = spamfuz.fuzzy_predict(40, 90)
    spam_score_fuzzy = spamfuz.fuzzy_predict(90, 20)
    spam_score_fuzzy = spamfuz.fuzzy_predict(90, 40)
    spam_score_fuzzy = spamfuz.fuzzy_predict(45, 70)
    spam_score_fuzzy = spamfuz.fuzzy_predict(70, 60)
    spam_score_fuzzy = spamfuz.fuzzy_predict(51, 90)
    exit(0)
import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime

from pandas import CategoricalDtype


def date_to_number_range(date):
    # Options for dates are:
    # 1. 1st of month -> month-year
    # 2. not 1st of month -> day-month
    if type(date) != datetime.datetime:
        return date

    if date.day == 1:
        return date.strftime("%m-%Y")
    else:
        return date.strftime("%d-%m")


def clean_up_data(raw_df):
    clean_df = raw_df
    clean_df.replace('?', np.nan, inplace=True)

    clean_df['inv-nodes'] = clean_df['inv-nodes'].apply(date_to_number_range)
    clean_df['tumor-size'] = clean_df['tumor-size'].apply(date_to_number_range)

    clean_df = clean_df.astype('category')

    # Fill in missing inv-nodes categories.
    inv_nodes_categories = CategoricalDtype(
        categories=['0-2', '03-05', '06-08', '09-11', '12-14', '15-17', '18-20', '21-23', '24-26'], ordered=True)
    clean_df['inv-nodes'] = clean_df['inv-nodes'].astype(inv_nodes_categories)

    return clean_df


def basic_stats(df):
    for col in df.columns:
        if col == 'Class':
            pass
        else:
            contingency_table = pd.crosstab(df['Class'], df[col])
            print(contingency_table)
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            print(f'Class vs {col}: chi2 = {chi2:.2f}, p-value = {p:.3f}')
            print('\n\n')



class LayerDense:
    def __init__(self, number_of_inputs, number_of_neurons ):
        self.weights = np.random.randn(number_of_inputs, number_of_neurons)
        self.bias = np.zeros((1, number_of_neurons))

    def forward(self, input):
        self.output = self.activation_ReLU(np.dot(input, self.weights) + self.bias)

    @staticmethod
    def activation_ReLU(input):
        return np.maximum(0, input)

    @staticmethod
    def softmax(inputs):
        """
        Subtraction of the max value in np.exp prevents overflow errors with large numbers.
        The subtraction means all exp values are less than 0, therefore all exp outputs are between 0 and 1.
        This does not affect the output after normalisation, so is purely there as a computing issue.
        :param inputs:
        :return:
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # Exponentiate the inputs
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Normalise the inputs based on sum of inputs.
        return probabilities

    @staticmethod
    def loss_categorical_cross_entropy(predictions, actual):
        """
        Uses one hot encoded vectors. This is where the "actual" or "correct" values given is what the neural net should give.
        i.e. actual = [[0, 1], [0, 1]]
            predicted [[0.9, 0.1], # Terrible prediction
                       [0.2, 0.8], # Good prediction

        :param predictions: output of softmax
        :param actual: array of the true values (array of 1 and 0s)
        :return: array of loss values for each input row.
        """
        number_of_samples = len(predictions)
        predictions = np.clip(predictions, 1e-7, 1-1e-7)  # clip 0 values from being inf, so no errors later.
        confidence = np.sum(predictions * actual, axis=1)  # Output 1-d array of activation of correct
        negative_log_probability = -np.log(confidence)
        return negative_log_probability


class NeuralNet:
    def __init__(self, input_df, number_of_neurons):
        self.df = input_df

    def df_preprocessing(self):
        """Function to convert a human readable df to a machine learnable df by converting everything to numners"""
        pass

    def preprocess_training_data(self, training_df):
        """
        This function should process the training data and store any features required in the class
        """
        pass
        # return processed_df

    def preprocess_test_data(self, test_df):
        pass
        # return processed_df


def main():
    X = np.random.randn(3, 4)
    layer1 = LayerDense(4, 5)
    layer2 = LayerDense(5, 2)

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)


    # pd.set_option('display.max_columns', None)
    #
    # df = pd.read_excel("./breast-cancer.xls")
    # df = clean_up_data(df)
    # print(df.describe())
    # print(len(df.columns))
    # Dont change
    # my_model = Module4_Model()
    # x_train_processed = my_model.preprocess_training_data(x_train)


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""
File: naive_bayes_classifier.py
Authors: Ben Krausz, Eva Ogbe

Uses a Naive Bayes Classifier to make predictions for a Bernoulli RV.
The probability estimates are determined using MLE first, then Laplace.

Usage:
    naive_bayes_classifier.py <train> <test>
where
    <train> is the name of the training file
    <test> is the name of the testing file
"""

import sys
from test_result import TestResult
from training_result import TrainingResult, OUTPUT_VALUES
from maximizer import Maximizer
from report_test_results import report_test_results

ESTIMATORS = {"MLE": 0, "Lapace": 1}

class NaiveBayesClassifier(object):
    def __init__(self, train_file, test_file):
        self.__train_file = train_file
        self.__test_file = test_file
    
    def train_mle(self):
        return self.train(0)
    
    def train_laplace(self):
        return self.train(1)
    
    def train(self, initial_occurrence_val):
        num_input_vars = None
        num_vectors = None
        training_result = None
        
        with open(self.__train_file, 'r') as f:
            for line in f:
                if num_input_vars is None:
                    num_input_vars = int(line)
                elif num_vectors is None:
                    num_vectors = int(line)
                    training_result = TrainingResult(num_input_vars, num_vectors, initial_occurrence_val)
                else:
                    x, y = line.rstrip().split(": ")
                    training_result.add_data(x.split(" "), y)
        
        training_result.train()
        return training_result
    
    def test(self, training_result):
        test_results = {output: TestResult() for output in OUTPUT_VALUES}
        
        with open(self.__test_file, 'r') as f:
            for lineno, line in enumerate(f):
                if lineno >= 2:
                    input_vector, y_hat = line.rstrip().split(": ")
                    x = input_vector.split(" ")
                    maximizer = Maximizer()
                    for y in OUTPUT_VALUES:
                        joint = training_result.calculate_joint(x, y)
                        maximizer.update(y, joint)
                    test_results[y_hat].tested += 1
                    if maximizer.y == y_hat: test_results[y_hat].correct += 1
        
        return test_results

if __name__ == "__main__":
    classifier = NaiveBayesClassifier(sys.argv[1], sys.argv[2])
    
    for name, initial_occurrence_val in ESTIMATORS.items():
        training_result = classifier.train(initial_occurrence_val)
        test_results = classifier.test(training_result)
        report_test_results(name, test_results)
        print()

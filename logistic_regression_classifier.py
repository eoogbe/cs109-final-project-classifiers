#!/usr/bin/env python
"""
File: logistic_regression_classifier.py
Authors: Ben Krausz, Eva Ogbe

Uses a Logistic Regression Classifier to make predictions for a Bernoulli RV.

Usage:
    logistic_regression_classifier.py <eta> <train> <test> [-d]
where
    <eta> is the learning rate
    <train> is the name of the training file
    <test> is the name of the testing file
    -d is a flag to show debug output
"""

import sys
from math import exp
from test_result import TestResult, OUTPUT_VALUES
from report_test_results import report_test_results

NUM_EPOCHS = 10000

class LogisticRegressionClassifier(object):
    def __init__(self, eta, train_file, test_file, debug):
        self.__eta = float(eta)
        self.__train_file = train_file
        self.__test_file = test_file
        self.__debug = debug
    
    def train(self):
        num_input_vars = None
        num_vectors = None
        betas = None
        output_vars = []
        input_vectors = []
        
        with open(self.__train_file, 'r') as f:
            for line in f:
                if num_input_vars is None:
                    num_input_vars = int(line)
                    betas = [0] * (num_input_vars + 1)
                elif num_vectors is None:
                    num_vectors = int(line)
                else:
                    input_vector, y = line.rstrip().split(": ")
                    
                    output_vars.append(int(y))
                    
                    x = [int(x_i) for x_i in input_vector.split(" ")]
                    x.insert(0, 1)
                    
                    input_vectors.append(x)
        
        for epoch in range(NUM_EPOCHS):
            gradients = [0] * (num_input_vars + 1)
            for i, input_vector in enumerate(input_vectors):
                y = output_vars[i]
                z = sum([betas[j] * x_j for j, x_j in enumerate(input_vector)])
                
                for k in range(num_input_vars + 1):
                    x_k = input_vector[k]
                    gradients[k] += x_k * (y - 1/(1 + exp(-z)))
            
            for k, gradient in enumerate(gradients):
                betas[k] += self.__eta * gradient
            
            if epoch % (NUM_EPOCHS / 100) == 0:
                print("\r{:.0f}% done".format(epoch / (NUM_EPOCHS / 100) + 1), end="",flush=True)
        
        print()
        
        if self.__debug:
            print("Betas:")
            for i, beta in enumerate(betas):
                print("{:2} | {}".format(i + 1, beta))
        
        return betas
    
    def test(self, betas):
        test_results = {output: TestResult() for output in OUTPUT_VALUES}
        
        with open(self.__test_file, 'r') as f:
            for lineno, line in enumerate(f):
                if lineno >= 2:
                    input_vector, y = line.rstrip().split(": ")
                    
                    x = [int(x_i) for x_i in input_vector.split(" ")]
                    x.insert(0, 1)
                    
                    z = sum([betas[j] * x_j for j, x_j in enumerate(x)])
                    p_y = 1 / (1 + exp(-z))
                    y_hat = "1" if p_y > .5 else "0"
                    test_results[y].tested += 1
                    if y_hat == y: test_results[y].correct += 1
        
        return test_results
    
if __name__ == "__main__":
    classifier = LogisticRegressionClassifier(sys.argv[1], sys.argv[2], sys.argv[3], len(sys.argv) >= 5)
    
    betas = classifier.train()
    test_results = classifier.test(betas)
    report_test_results(test_results)

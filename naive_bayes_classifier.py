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

INPUT_VALUES = ["0", "1"]
OUTPUT_VALUES = ["0", "1"]
NUM_INPUT_OUTPUT_COMBINATIONS = len(INPUT_VALUES) * len(OUTPUT_VALUES)

def create_initial_tables(num_tables, initial_val):
    return [{x: {y: initial_val for y in OUTPUT_VALUES} for x in INPUT_VALUES} for i in range(num_tables)]

def train(filename, initial_occurrence_val):
    num_input_vars = None
    num_vectors = None
    occurrence_tables = None
    y_occurrences = {y: 0 for y in OUTPUT_VALUES}
    
    with open(filename, 'r') as f:
        for line in f:
            if num_input_vars is None:
                num_input_vars = int(line)
                occurrence_tables = create_initial_tables(num_input_vars, initial_occurrence_val)
            elif num_vectors is None:
                num_vectors = int(line)
            else:
                x, y = line.rstrip().split(": ")
                y_occurrences[y] += 1
                for i, x_i in enumerate(x.split(" ")):
                    occurrence_tables[i][x_i][y] += 1
    
    total_occurrences = num_vectors + initial_occurrence_val * NUM_INPUT_OUTPUT_COMBINATIONS
    prob_estimates = [{x: {y: joint_occurrences / total_occurrences for y, joint_occurrences in class_occurrences.items()} for x, class_occurrences in table.items()} for i, table in enumerate(occurrence_tables)]
    priors = {y: occurrence / total_occurrences for y, occurrence in y_occurrences.items()}
    
    for i, table in enumerate(prob_estimates):
        print("{0} | {1}".format(i + 1, table["1"]["1"]))
    
    return (prob_estimates, priors)

def train_mle(filename):
    return train(filename, 0)

def test(filename, prob_estimates, priors):
    num_input_vars = None
    num_vectors = None
    test_results = None
    
    with open(filename, 'r') as f:
        for line in f:
            if num_input_vars is None:
                num_input_vars = int(line)
            elif num_vectors is None:
                num_vectors = int(line)
                test_results = {output: [0, 0] for output in OUTPUT_VALUES}
            else:
                x, y_hat = line.rstrip().split(": ")
                maximizer = None
                for y in OUTPUT_VALUES:
                    joint = priors[y]
                    for i, x_i in enumerate(x.split(" ")):
                        joint *= prob_estimates[i][x_i][y]
                    if maximizer is None or joint > maximizer[1]:
                        maximizer = (y, joint)
                test_results[y_hat][0] += 1
                if maximizer[0] == y_hat: test_results[y_hat][1] += 1
    
    return test_results

if __name__ == "__main__":
    mle_prob_estimates, mle_priors = train_mle(sys.argv[1])
    mle_test_results = test(sys.argv[2], mle_prob_estimates, mle_priors)
    
    for output_class, class_result in mle_test_results.items():
        print("Class {0}: tested: {1}, correctly classified {2}".format(output_class, *class_result))
    
    total_tested = sum([x[0] for x in mle_test_results.values()])
    total_correct = sum([x[1] for x in mle_test_results.values()])
    print("Overall: tested {0}, correctly classified {1}".format(total_tested, total_correct))
    print("Accuracy = {0}".format(total_correct / total_tested))

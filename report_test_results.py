def report_test_results(test_results):
    for output_class, class_result in test_results.items():
        print("Class {0}: tested: {1}, correctly classified {2}".format(output_class, class_result.tested, class_result.correct))
    
    class_results = test_results.values()
    total_tested = sum([result.tested for result in class_results])
    total_correct = sum([result.correct for result in class_results])
    
    print("Overall: tested {0}, correctly classified {1}".format(total_tested, total_correct))
    print("Accuracy = {0}".format(total_correct / total_tested))

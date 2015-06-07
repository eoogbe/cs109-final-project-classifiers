INPUT_VALUES = ["0", "1"]
OUTPUT_VALUES = ["0", "1"]

class TrainingResult(object):
    def __init__(self, num_input_vars, num_vectors, initial_occurrence_val):
        self.__num_input_vars = num_input_vars
        self.__num_vectors = num_vectors
        self.__initial_occurrence_val = initial_occurrence_val
        self.__occurrence_tables = self.__init_tables()
        self.__y_occurrences = self.__init_y_occurrences()
    
    def train(self):
        total_occurrences = self.__calculate_total_occurrences()
        self.__prob_estimates = [{x: {y: joint_occurrences / total_occurrences for y, joint_occurrences in class_occurrences.items()} for x, class_occurrences in table.items()} for i, table in enumerate(self.__occurrence_tables)]
        self.__priors = {y: occurrence / total_occurrences for y, occurrence in self.__y_occurrences.items()}
    
    def add_data(self, x, y):
        self.__y_occurrences[y] += 1
        for i, x_i in enumerate(x):
            self.__occurrence_tables[i][x_i][y] += 1
    
    def calculate_joint(self, x, y):
        prior = self.__priors[y]
        joint = prior
        
        for i, x_i in enumerate(x):
            joint *= self.__prob_estimates[i][x_i][y] / prior
        
        return joint
    
    def __repr__(self):
        return "\n".join(["{0} | {1}".format(i + 1, table["1"]["1"] / self.__priors["1"]) for i, table in enumerate(self.__prob_estimates)])
    
    def __init_tables(self):
        return [{x: {y: self.__initial_occurrence_val for y in OUTPUT_VALUES} for x in INPUT_VALUES} for i in range(self.__num_input_vars)]
    
    def __init_y_occurrences(self):
        return {y: self.__initial_occurrence_val * len(INPUT_VALUES) for y in OUTPUT_VALUES}
    
    def __calculate_total_occurrences(self):
        return self.__num_vectors + self.__initial_occurrence_val * len(INPUT_VALUES) * len(OUTPUT_VALUES)
    
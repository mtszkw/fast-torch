import numpy as np
# https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
from timeit import default_timer as timer


class PythonInference:
    def __init__(self, model):
        self.model = model

    def check_inference_time(self, tokenized_input: dict):
        t = timer()
        output = self.model(**tokenized_input)
        elapsed_time = timer()-t
        return elapsed_time

    def check_inference_time_all(self, tokenized_inputs: list, num_experiments=1):
        time_measurements = []
        for i in range(num_experiments):
            time_measurements.append([self.check_inference_time(x) for x in tokenized_inputs])
        return np.array(time_measurements)


class OnxxInference:
    def __init__(self, session):
        self.session = session

    def check_inference_time(self, tokenized_input):
        t = timer()
        output = self.session.run(None, tokenized_input)
        elapsed_time = timer()-t
        return elapsed_time
    
    def check_inference_time_all(self, tokenized_inputs: list, num_experiments=1):
        time_measurements = []
        for i in range(num_experiments):
            time_measurements.append(
                [self.check_inference_time({name: np.atleast_2d(value) for name, value in x.items()}) for x in tokenized_inputs])
        return np.array(time_measurements)
        
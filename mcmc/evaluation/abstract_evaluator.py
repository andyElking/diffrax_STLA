class AbstractEvaluator:
    def eval(self, samples, aux_output, ground_truth, config, model):
        raise NotImplementedError

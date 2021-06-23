class LabelModel:
    def __init__(self):
        raise NotImplementedError
    
    def train(self, rename_me1, rename_me2):
        # Feel free to add more inputs as needed
        raise NotImplementedError
        
    def predict(self, unlabeled_vote_matrix):
        # This should aggregate the votes from various taglets and output labels for the unlabeled data
        # Feel free to add more inputs as needed
        raise NotImplementedError
    
    def _helper_func(self):
        raise NotImplementedError
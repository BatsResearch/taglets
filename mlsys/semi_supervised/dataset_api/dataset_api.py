class DatasetAPI:
    def __init__(self):
        pass
    
    def get_num_checkpoints(self):
        raise NotImplementedError
    
    def get_class_names(self):
        raise NotImplementedError
    
    def get_labeled_dataset(self, checkpoint_num):
        raise NotImplementedError
        
    def get_unlabeled_dataset(self, checkpoint_num, train):
        raise NotImplementedError
        
    def get_test_dataset(self):
        raise NotImplementedError
        
    def get_test_labels(self):
        raise NotImplementedError
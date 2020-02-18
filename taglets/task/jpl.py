import logging
import requests
from ..modules import RandomActiveLearning, LeastConfidenceActiveLearning
from ..task import Task
from ..controller import Controller

log = logging.getLogger(__name__)


class JPL:
    """
    A class to interact with JPL-like APIs.
    """
    def __init__(self):
        """
        Create a new JPL object.
        """
        self.secret = 'a5aed2a8-db80-4b22-bf72-11f2d0765572'
        self.url = 'http://myserviceloadbalancer-679310346.us-east-1.elb.amazonaws.com'
        self.session_token = ''
        self.data_type = 'sample'   # Sample or full

    def get_available_tasks(self):
        """
        Get all available tasks.
        :return: A list of tasks (problems)
        """
        headers = {'user_secret': self.secret}
        r = requests.get(self.url + "/list_tasks", headers=headers)
        return r.json()['tasks']

    def get_task_metadata(self, task_name):
        """
        Get metadata about a task.
        :param task_name: The name of the task (problem)
        :return: The task metadata
        """
        headers = {'user_secret': self.secret}
        r = requests.get(self.url + "/task_metadata/" + task_name, headers=headers)
        return r.json()['task_metadata']

    def create_session(self, task_name):
        """
        Create a new session.
        :param task_name: The name of the task (problem
        :return: None
        """
        headers = {'user_secret': self.secret}
        r = requests.get(self.url + "/auth/get_session_token/" + self.data_type + "/" + task_name, headers=headers)
        session_token = r.json()['session_token']
        self.session_token = session_token

    def get_session_status(self):
        """
        Get the session status.
        :return: The session status
        """
        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.get(self.url + "/session_status", headers=headers)
        return r.json()['Session_Status']

    def get_seed_labels(self):
        """
        Get seed labels.
        :return: A list of lists with name and label e.g., ['1.png', '2'], ['2.png', '7'], etc.
        """
        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.get(self.url + "/seed_labels", headers=headers)
        labels = r.json()['Labels']
        seed_labels = []
        for image in labels:
            seed_labels.append([image["id"], image["class"]])
        return seed_labels

    def request_label(self, query):
        """
        Get labels of requested examples.

        :param query: A dictionary where the key is 'example_ids', and the value is a list of filenames.
        For example:
        query = {
            'example_ids': [
                '45781.png',
                '40214.png',
                '49851.png',
            ]
        }

        :return: A list of lists containing the name of the example and its label.
        For example:
         [['45781.png', '3'], ['40214.png', '7'], ['49851.png', '8']]
        """
        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.post(self.url + "/query_labels", json=query, headers=headers)
        return r.json()['Labels']

    def submit_prediction(self, predictions):
        """
        Submit predictions on test images.

        :param predictions: A dictionary containing test image names and corresponding labels.
        For example:
        predictions = {'id': ['6831.png', '1186.png', '8149.png', '4773.png', '3752.png'],
                       'label': ['9', '6', '9', '2', '10']}
        :return: The session status after submitting prediction
        """

        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.post(self.url + "/submit_predictions", json={'predictions': predictions}, headers=headers)
        return r.json()
            

class JPLRunner:
    def __init__(self, use_gpu=False, testing=False, data_type='sample'):
        self.api = JPL()
        self.api.data_type = data_type
        self.task, self.num_base_checkpoints, self.num_adapt_checkpoints = self.api.get_task()
        self.random_active_learning = RandomActiveLearning()
        self.confidence_active_learning = LeastConfidenceActiveLearning()
        self.controller = Controller()
        
        self.use_gpu = use_gpu
        self.testing = testing
        
    def get_task(self):
        task_names = self.api.get_available_tasks()
        task_name = task_names[0]  # Image classification task
        self.api.create_session(task_name)
        task_metadata = self.api.get_task_metadata(task_name)
    
        num_base_checkpoints = len(task_metadata['base_label_budget'])
        num_adapt_checkpoints = len(task_metadata['adaptation_label_budget'])
    
        task = Task(task_name, task_metadata)
        session_status = self.api.get_session_status()
        current_dataset = session_status['current_dataset']
        task.classes = current_dataset['classes']
        task.number_of_channels = current_dataset['number_of_channels']
    
        task.unlabeled_image_path = "./sql_data/MNIST/" + self.api.data_type + "/train"
        task.evaluation_image_path = "./sql_data/MNIST/" + self.api.data_type + "test"  # Should be updated later
        task.phase = session_status['pair_stage']
        if session_status['pair_stage'] == 'adaptation':
            task.labeled_images = []
            task.pretrained = task_metadata['adaptation_can_use_pretrained_model']
        elif session_status['pair_stage'] == 'base':
            task.labeled_images = self.api.get_seed_labels()
            task.pretrained = task_metadata['base_can_use_pretrained_model']
        return task, num_base_checkpoints, num_adapt_checkpoints

    def update_task(self):
        task_metadata = self.api.get_task_metadata(self.task.name)
        session_status = self.api.get_session_status()
        current_dataset = session_status['current_dataset']
        self.task.classes = current_dataset['classes']
        self.task.number_of_channels = current_dataset['number_of_channels']
    
        self.task.unlabeled_image_path = "./sql_data/MNIST/" + self.api.data_type + "/train"
        self.task.evaluation_image_path = "./sql_data/MNIST/" + self.api.data_type + "/test"  # Should be updated later
        self.task.phase = session_status['pair_stage']
        if session_status['pair_stage'] == 'adaptation':
            self.task.labeled_images = []
            self.task.pretrained = task_metadata['adaptation_can_use_pretrained_model']
        elif session_status['pair_stage'] == 'base':
            self.task.labeled_images = self.api.get_seed_labels()
            self.task.pretrained = task_metadata['base_can_use_pretrained_model']

    def run_checkpoints(self):
        self.run_checkpoints_base()
        self.run_checkpoints_adapt()

    def run_checkpoints_base(self):
        self.update_task()
        for i in range(self.num_base_checkpoints):
            self.run_one_checkpoint("Base", i)
    
    def run_checkpoints_adapt(self):
        self.update_task()
        for i in range(self.num_base_checkpoints):
            self.run_one_checkpoint("Adapt", i)
    
    def run_one_checkpoint(self, phase, checkpoint_num):
        session_status = self.api.get_session_status()
        log.info('------------------------------------------------------------')
        log.info('--------------------{} Checkpoint: {}'.format(phase, checkpoint_num)+'---------------------')
        log.info('------------------------------------------------------------')

        available_budget = self.get_available_budget()
        unlabeled_image_names = self.task.get_unlabeled_image_names()
        log.info('number of unlabeled data: {}'.format(len(unlabeled_image_names)))
        if checkpoint_num == 0:
            candidates = self.random_active_learning.find_candidates(available_budget, unlabeled_image_names)
        else:
            candidates = self.confidence_active_learning.find_candidates(available_budget, unlabeled_image_names)
        self.request_labels(candidates)
        predictions = self.controller.get_predictions(self.task) # will get EndModel instead of predictions
        self.submit_predictions(predictions)

    def get_available_budget(self):
        session_status = self.api.get_session_status()
        available_budget = session_status['budget_left_until_checkpoint']

        if self.testing:
            available_budget = available_budget // 10
        return available_budget

    def request_labels(self, examples):
        query = {'example_ids': examples}
        labeled_images = self.api.request_label(query)
        self.task.add_labeled_images(labeled_images)
        log.info("New labeled images:", len(labeled_images))
        log.info("Total labeled images:", len(self.task.labeled_images))

    def submit_predictions(self, predictions):
        submit_status = self.api.submit_prediction(predictions)
        session_status = self.api.get_session_status()
        log.info("Checkpoint scores", session_status['checkpoint_scores'])
        log.info("Phase:", session_status['pair_stage'])


def main():
    # TODO: Make CLI
    runner = JPLRunner('sample')
    runner.run_checkpoints()


if __name__ == "__main__":
    main()

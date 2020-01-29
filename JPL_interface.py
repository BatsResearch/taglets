import requests


class JPL:
    """
    A class to interact with JPL.
    """
    def __init__(self):
        """
        Create a new JPL object.
        """
        self.secret = '97edc318-11de-4e48-85cb-ef515bb24093'
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

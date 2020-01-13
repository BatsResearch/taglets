import requests
import pandas as pd
from pathlib import Path
import random


class JPL():
    def __init__(self):
        self.secret = '97edc318-11de-4e48-85cb-ef515bb24093'
        self.url = 'http://myserviceloadbalancer-679310346.us-east-1.elb.amazonaws.com'
        self.session_token = ''
        self.data_type = 'sample' #sample or full

    def get_available_tasks(self):
        """List all of the available tasks (or problems). It returns a list of tasks"""

        headers = {'user_secret': self.secret}
        r = requests.get(self.url + "/list_tasks", headers= headers)
        return r.json()['tasks']

    def get_task_metadata(self, task_name):
        """return task metadata"""

        headers = {'user_secret': self.secret}
        r = requests.get(self.url + "/task_metadata/"+task_name, headers=headers)
        return r.json()['task_metadata']

    def create_session(self, task_name):
        """Create a New Session"""

        headers = {'user_secret': self.secret}
        r = requests.get(self.url + "/auth/get_session_token/" + self.data_type + "/" + task_name, headers= headers)
        session_token = r.json()['session_token']
        self.session_token = session_token

    def get_session_status(self):
        """return session status"""

        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.get(self.url + "/session_status", headers=headers)
        return r.json()['Session_Status']

    def get_seed_labels(self):
        """return seed labels

        It returns a dictionary with the key as the name of image, and value as label. For example:

        {'id': '56847.png', 'label': '2'},
          {'id': '45781.png', 'label': '3'},
          {'id': '40214.png', 'label': '7'},
          {'id': '49851.png', 'label': '8'},
          {'id': '46024.png', 'label': '6'},
          {'id': '13748.png', 'label': '1'},
          {'id': '13247.png', 'label': '9'},
          {'id': '39791.png', 'label': '4'},
          {'id': '37059.png', 'label': '0'},
          {'id': '46244.png', 'label': '5'}

        """

        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.get(self.url + "/seed_labels", headers=headers)
        return r.json()['Labels'][0]

    def request_label(self, query):
        """return the label of requested examples

        :param examples: a dictionary; the key is 'example_ids', and the value is a list of images. For example:

        query = {
            'example_ids': [
                '45781.png',
                '40214.png',
                '49851.png',
            ]
        }

        :return: list of lists, with the name of example and its label. For example:
         [['45781.png', '3'], ['40214.png', '7'], ['49851.png', '8']]
        """

        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.post(self.url + "/query_labels", json=query, headers=headers)

        return r.json()['Labels']

    def submit_prediction(self, predictions):
        """submit predictions on test images

        :param predictions: a dictionary of test images and correspondig labels. For example:
        predictions = {'id': {0: '6831.png', 1: '1186.png', 2: '8149.png', 3: '4773.png', 4: '3752.png'},
                    'label':{'label': {0: '9',  1: '6',  2: '9',  3: '2',  4: '10'}

        :return: session status after submitting prediction
        """

        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.post(self.url+"/submit_predictions", json={'predictions': predictions}, headers=headers)
        return r.jason()



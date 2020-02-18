import io
import os
import requests
import zipfile


def check_test_data():
    if not os.path.isdir("mnist_sample"):
        r = requests.get("http://cs.brown.edu/people/sbach/mnist_sample.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()

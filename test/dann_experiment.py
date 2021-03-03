from taglets.controller import Controller
from taglets.scads import Scads
from taglets.scads.create.scads_classes import Image
from taglets.task import Task
from taglets.data import CustomDataset

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import unittest

EXTERNAL_DATA = "/lwll/external"
TEST_DATA = "/home/ubuntu/jeff/taglets/test/test_data/scads"
DOMAIN_NET_PAINTING_DB_PATH = os.path.join(TEST_DATA, "domainnet_painting.db")
DOMAIN_NET_QUICKDRAW_DB_PATH = os.path.join(TEST_DATA, "domainnet_quickdraw.db")
DOMAIN_NET_CLIPART_DB_PATH = os.path.join(TEST_DATA, "domainnet_clipart.db")
DOMAIN_NET_REAL_DB_PATH = os.path.join(TEST_DATA, "domainnet_real.db")
DOMAIN_NET_REAL_DB_PATH1 = os.path.join(TEST_DATA, "domainnet_real1.db")
DOMAIN_NET_REAL_DB_PATH2 = os.path.join(TEST_DATA, "domainnet_real2.db")
EMBEDDING_PATH = os.path.join(TEST_DATA, "test_embedding.h5")
DOMAIN_NET_CLASSES = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil',
               'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat',
               'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench',
               'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang',
               'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket',
               'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera',
               'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan',
               'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud',
               'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile',
               'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut',
               'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant',
               'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant',
               'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower',
               'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee',
               'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones',
               'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital',
               'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream',
               'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf',
               'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
               'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
               'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom',
               'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush',
               'paint_can', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut',
               'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow',
               'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
               'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control',
               'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw',
               'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark',
               'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
               'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat',
               'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
               'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine',
               'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 'table', 'teapot', 'teddy-bear',
               'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China',
               'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
               'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt',
               'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide',
               'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

for i in range(len(DOMAIN_NET_CLASSES)):
    DOMAIN_NET_CLASSES[i] = "/c/en/" + DOMAIN_NET_CLASSES[i]

class HiddenLabelDataset(Dataset):
    """
    Wraps a labeled dataset so that it appears unlabeled
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img

    def __len__(self):
        return len(self.dataset)


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        classes = [
            '/c/en/cake',
            '/c/en/cat',
            '/c/en/mountain',
            '/c/en/sheep',
            '/c/en/elephant',
            '/c/en/bread',
            '/c/en/garden',
            '/c/en/mushroom',
            '/c/en/helicopter',
            '/c/en/bird'
        ]
        # Build ScadsEmbedding File
        arr = []
        for i in range(len(classes)):
            l = [0.0] * len(classes)
            l[i] = 1.0
            arr.append(l)
        arr = np.asarray(arr)
        df = pd.DataFrame(arr, index=classes, dtype='f')
        df.to_hdf(EMBEDDING_PATH, key='mat', mode='w')

    def test_domain_net(self):
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        preprocess = transforms.Compose([
            transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std)
        ])

        Scads.open(DOMAIN_NET_REAL_DB_PATH)
        images = Scads.session.query(Image)
        # label_counts = {}
        # for image in images:
        #     id = image.node.conceptnet_id
        #     if id not in label_counts:
        #         label_counts[id] = 0
        #     label_counts[id] += 1
        # classes = set(sorted(label_counts, key=lambda x: label_counts[x], reverse=True)[:10])
        classes = {
            '/c/en/cake',
            '/c/en/cat',
            '/c/en/mountain',
            '/c/en/sheep',
            '/c/en/elephant',
            '/c/en/bread',
            '/c/en/garden',
            '/c/en/mushroom',
            '/c/en/helicopter',
            '/c/en/bird'
        }

        # ids = []
        filepaths = []
        labels = []
        num_labels = 0
        id_to_label = {}
        for image in images:
            id = image.node.conceptnet_id
            if id in classes:
                if id not in id_to_label:
                    id_to_label[id] = num_labels
                    num_labels += 1
                labels.append(id_to_label[id])
                filepaths.append(os.path.join(EXTERNAL_DATA, image.path))
                # ids.append((id, image.id))
        # for id in classes:
        #     median = sorted(list(filter(lambda x: x[0] == id, ids)))
        #     print(len(median))
        #     print(id, median[len(median) // 2])
        data = list(zip(filepaths, labels))
        np.random.shuffle(data)
        filepaths, labels = zip(*data)

        global DOMAIN_NET_CLASSES
        DOMAIN_NET_CLASSES = list(filter(lambda x: x in classes, DOMAIN_NET_CLASSES))
        print(DOMAIN_NET_CLASSES)
        domainnet_real = CustomDataset(filepaths=filepaths, labels=labels, transform=preprocess)

        # Creates task
        labeled = Subset(domainnet_real, [i for i in range(100)])
        unlabeled = HiddenLabelDataset(Subset(domainnet_real, [i for i in range(100, 1000)]))
        val = Subset(domainnet_real, [i for i in range(1000, 1100)])

        task = Task(
            "domain-net-test", DOMAIN_NET_CLASSES, (256, 256), labeled, unlabeled, val, scads_path=DOMAIN_NET_QUICKDRAW_DB_PATH,
            scads_embedding_path=EMBEDDING_PATH
        )

        # Executes task
        controller = Controller(task)
        end_model = controller.train_end_model()

        # Evaluates end model
        domainnet_real_test = Subset(domainnet_real, [i for i in range(1100, 2100)])
        accuracy = end_model.evaluate(domainnet_real_test)
        print("DomainNet Real Accuracy:", accuracy)
        self.assertGreater(accuracy, .9)


if __name__ == "__main__":
    unittest.main()

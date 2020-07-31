import cv2
import numpy as np
import pickle

from torch.utils.data import Dataset
from Utils.dataloader_utils import *

class VGGFace2Dataset_Online(Dataset):

    def __init__(self, data_dir, identities, samples, batches_per_epoch = 32, face_dict_dir = None, transforms = None):
        
        self.data_dir = data_dir
        self.transform = transforms
        self.identities = identities
        self.samples = samples
        self.batches_per_epoch = batches_per_epoch
        if face_dict_dir is None:
            self.face_dict = make_dict(data_dir)
        else:
            if os.path.isfile(face_dict_dir):
                pkl = open(face_dict_dir, 'rb')
                self.face_dict = pickle.load(pkl)
                pkl.close()
            else:
                self.face_dict = make_dict(data_dir)


    def __getitem__(self, idx):
        
        batch_sample_dict = dict()
        batch_sample = np.empty(shape = (self.identities * self.samples, 3, 220, 220))
        identities = np.random.choice(list(self.face_dict.keys()), self.identities, replace = False)
        
        for identity in identities:
            samples = self.face_dict[identity]
            img_num = 0
            while img_num < self.samples:
                sample = np.random.choice(samples, 1, replace = False)
                samples.remove(sample)
                img_path = self.data_dir + '/' + str(identity) + '/' + str(sample[0])
                if check_img(img_path):
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    img_processed = extract_face(img)
                    if self.transform:
                        img_processed = self.transform(img_processed)
                    batch_sample[img_num] = np.resize(img_processed, (3, 220, 220))
                    img_num += 1

        return batch_sample


    def __len__(self):
        
        return self.batches_per_epoch
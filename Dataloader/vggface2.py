import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import extract_face 

class VGGFace2Dataset(Dataset):
    
    def __init__(self, training_triplets_path, data_dir, transform = None):
        self.transform = transform
        self.training_triplets = np.load(training_triplets_path)
        self.data_dir = data_dir
        
    def __getitem__(self, idx):
        _, _, _, _, _, anc_path, pos_path, neg_path = self.training_triplets[idx]
        print(anc_path)
        print(pos_path)
        print(neg_path)
        _, anc_img = extract_face(cv2.cvtColor(cv2.imread(self.data_dir + '/' + anc_path), cv2.COLOR_BGR2RGB))
        _, pos_img = extract_face(cv2.cvtColor(cv2.imread(self.data_dir + '/' + pos_path), cv2.COLOR_BGR2RGB))
        _, neg_img = extract_face(cv2.cvtColor(cv2.imread(self.data_dir + '/' + neg_path), cv2.COLOR_BGR2RGB))
        
        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample
    
    def __len__(self):
        return len(self.training_triplets)

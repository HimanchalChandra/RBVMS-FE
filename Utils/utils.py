import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from Models.inception_resnet_v1 import InceptionResnetV1
from Utils.dataloader_utils import *
from torch.nn.modules.distance import PairwiseDistance


model = InceptionResnetV1().eval()

l2_dist = PairwiseDistance(2)


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
    )
])


def generate_embedding(img_path):

	flag = 0
	output = 0
	if check_img(img_path):
		flag = 1
		img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
		img_processed = extract_face(img)
		img_processed = data_transforms(img_processed)
		sample = np.resize(img_processed, (3, 220, 220))
		print(img_processed.shape)
		output = model(img_processed.view(-1,3,220,220))

	return output, flag


def calc_distance(embedding_1, embedding_2):
	embedding_1 = torch.tensor(embedding_1)
	embedding_2 = torch.tensor(embedding_2)
	return l2_dist.forward(embedding_1, embedding_2)



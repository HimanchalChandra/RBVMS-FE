from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance
from models.recog_net import Recog_Net
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from dataloader.vggface2 import VGGFace2Dataset
from utils.generate_triplets import generate_triplets
from utils.triplet_loss import TripletLoss
import torch
import numpy as np
import os


data_directory = '/media/ank99/Elements/Datasets/vggface_2/Data/vggface2_train/train'

total_triplets = 50

_, triplets_filename = generate_triplets(data_directory, total_triplets)
triplets_path = os.getcwd() + '/data/' + triplets_filename

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
        )
    ])

train_dataloader = DataLoader(
        dataset = VGGFace2Dataset(
            training_triplets_path = triplets_path,
            data_dir = data_directory,
            transform = data_transforms
            ),
        batch_size = 1,
        shuffle = False
        )

net = Recog_Net()
net.cuda()
margin = 0.05
l2_distance = PairwiseDistance(2).cuda()

optimizer_model = SGD(net.parameters(), lr=0.1)

for epoch in range(0,2):
    triplet_loss_sum = 0
    num_valid_training_triplets = 0
    progress_bar = enumerate(tqdm(train_dataloader))
    for batch_idx, (batch_sample) in progress_bar:
        anc_image = batch_sample['anc_img'].view(-1, 3, 220, 220).cuda()
        pos_image = batch_sample['pos_img'].view(-1, 3, 220, 220).cuda()
        neg_image = batch_sample['neg_img'].view(-1, 3, 220, 220).cuda()

        anc_embedding = net(anc_image)
        pos_embedding = net(pos_image)
        neg_embedding = net(neg_image)

        
        pos_dist = l2_distance.forward(anc_embedding, pos_embedding)
        neg_dist = l2_distance.forward(anc_embedding, neg_embedding)
        print(pos_dist)
        print(neg_dist)
        print('\n')

        allcd = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
        hard_triplets = np.where(allcd == 1)
        anc_hard_embedding = anc_embedding[hard_triplets].cuda()
        pos_hard_embedding = pos_embedding[hard_triplets].cuda()
        neg_hard_embedding = neg_embedding[hard_triplets].cuda()

        triplet_loss = TripletLoss(margin=margin).forward(
                    anchor=anc_hard_embedding,
                    positive=pos_hard_embedding,
                    negative=neg_hard_embedding
                ).cuda()

        triplet_loss_sum += triplet_loss.item()
        num_valid_training_triplets += len(anc_hard_embedding)

        optimizer_model.zero_grad()
        triplet_loss.backward()
        optimizer_model.step()

    avg_triplet_loss = 0 if (num_valid_training_triplets == 0) else triplet_loss_sum / num_valid_training_triplets

    print('Epoch {}:\tAverage Triplet Loss: {:.4f}\tNumber of valid training triplets in epoch: {}'.format(
            epoch+1,
            avg_triplet_loss,
            num_valid_training_triplets
        )
    )

torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer_model.state_dict(),
    'avg_triplet_loss': avg_triplet_loss,
    'valid_training_triplets': num_valid_training_triplets 
    }, './train_checkpoints/' + 'checkpoint_' + str(total_triplets) + '_' + str(epoch) + '_' + str(num_valid_training_triplets)  + '.tar')

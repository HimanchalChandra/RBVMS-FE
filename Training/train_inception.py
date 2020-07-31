import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Models.inception_resnet_v1 import InceptionResnetV1
from Dataloader.online_mining import VGGFace2Dataset_Online
from Utils.train_utils import *
from torch.optim import SGD
import torch.nn as nn
import torch
from tqdm import tqdm


epochs = 1
learning_rate = 0.1
margin = 0.002

batches_per_epoch = 1

identities_per_batch = 3
samples_per_class = 3

face_dict_directory = os.getcwd() + '/Data/face_dict.pkl'
data_directory = '/media/ank99/New Volume/Datasets/vggface2_train/train'

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
    )
])

train_dataloader = DataLoader(
    dataset = VGGFace2Dataset_Online(
        identities = identities_per_batch,
        samples = samples_per_class,
        batches_per_epoch = batches_per_epoch,
        face_dict_dir = face_dict_directory,
        data_dir = data_directory,
        transforms = data_transforms
    ),
    batch_size = 1,
    shuffle = False
)


net = InceptionResnetV1()
net.cuda()

Triplet_Loss = nn.TripletMarginLoss(margin=margin, p=2)
optimizer = SGD(net.parameters(), lr=learning_rate)



for epoch in range(0, epochs):
	triplet_loss_sum = 0
	batches = enumerate(tqdm(train_dataloader))

	for batch_idx, (batch_sample) in batches:
		batch_sample = batch_sample.view(-1, 3, 220, 220).float().cuda()
		embeddings = net(batch_sample)
		anchors = embeddings
		positives, negatives = get_hardest_pos_neg(samples_per_class, identities_per_batch, anchors)
		output = Triplet_Loss(anchors, positives, negatives)
		triplet_loss_sum += output
		optimizer.zero_grad()
		output.backward()
		optimizer.step()
		print(torch.cuda.max_memory_allocated(device=0))
		print(output)
		with open('./Logs/log_triplet_new.txt', 'a') as f:
			val_list = [
				epoch+1,
				batch_idx,
				float(output),
				float(triplet_loss_sum)
			]
			log = '\t'.join(str(value) for value in val_list)
			f.writelines(log + '\n')

	avg_triplet_loss = triplet_loss_sum / batches_per_epoch

	with open('./Logs/log_triplet_new.txt', 'a') as f:
		val_list = [
			'FINAL',
			epoch+1,
			float(avg_triplet_loss)
		]
		log = '\t'.join(str(value) for value in val_list)
		f.writelines(log + '\n')

	if ((epoch+1) == 56):
		torch.save({
		'epoch': epoch+1,
		'model_state_dict': net.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'avg_triplet_loss': avg_triplet_loss
		}, './Train_Checkpoints/' + 'checkpoint_' + str(epoch+1) + '_' + str(round(float(avg_triplet_loss), 4)) + '.tar')


	print('Epoch {}:\tAverage Triplet Loss: {:.7f}\t'.format(
			epoch+1,
			avg_triplet_loss
			)
	)

torch.save({
	'epoch': epoch+1,
	'model_state_dict': net.state_dict(),
	'optimizer_state_dict': optimizer.state_dict(),
	'avg_triplet_loss': avg_triplet_loss
	}, './Train_Checkpoints/' + 'checkpoint_' + str(epoch+1) + '_' + str(round(float(avg_triplet_loss), 4)) + '.tar')


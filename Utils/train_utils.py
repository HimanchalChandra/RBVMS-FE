import torch
import numpy as np
from torch.nn.modules.distance import PairwiseDistance


def get_masks(samples_per_class, identities_per_batch):

    ones = np.ones((samples_per_class,samples_per_class))
    zeros = np.zeros((samples_per_class,samples_per_class))
    eye = np.eye(identities_per_batch, identities_per_batch)
    pos_mask = np.empty((0, samples_per_class*identities_per_batch))
    neg_mask = np.empty((0, samples_per_class*identities_per_batch))

    for i in range(0, identities_per_batch):
        pos_mask_unit = np.empty((samples_per_class,0))
        neg_mask_unit = np.empty((samples_per_class,0))
        for t in eye[i]:
            if t == 1:
                pos_mask_unit = np.concatenate((pos_mask_unit, ones), axis = 1)
                neg_mask_unit = np.concatenate((neg_mask_unit, zeros), axis = 1)
            else:
                pos_mask_unit = np.concatenate((pos_mask_unit, zeros), axis = 1)
                neg_mask_unit = np.concatenate((neg_mask_unit, ones), axis = 1)
        pos_mask = np.concatenate((pos_mask, pos_mask_unit), axis = 0)
        neg_mask = np.concatenate((neg_mask, neg_mask_unit), axis = 0)
    
    return pos_mask, neg_mask


def get_hardest_pos_neg(samples_per_class, identities_per_batch, embeddings):

    l2_dist = PairwiseDistance(2).cuda()
    tot_faces = samples_per_class*identities_per_batch
    dists = []

    for index, embedding in enumerate(embeddings):
        dists.append(list(l2_dist.forward(embedding, embeddings)))
    
    pos_mask, neg_mask = get_masks(samples_per_class, identities_per_batch)
    pos_mask = torch.tensor(pos_mask).cuda()
    neg_mask = torch.tensor(neg_mask).cuda()
    dists = torch.tensor(dists).cuda()
    pos_dists = torch.mul(dists, pos_mask)
    hardest_pos_dists = torch.max(pos_dists, 1)
    neg_dists = torch.mul(dists, neg_mask)
    neg_dists_nonzero = neg_dists[torch.nonzero(neg_dists, as_tuple=True)]
    neg_dists_nonzero_shape = (tot_faces, tot_faces-samples_per_class)
    neg_dists_nonzero = neg_dists_nonzero.view(neg_dists_nonzero_shape[0], neg_dists_nonzero_shape[1])
    hardest_neg_dists = torch.min(neg_dists_nonzero, 1)

    hardest_pos_embeddings = embeddings[hardest_pos_dists[1]]
    hardest_neg_embeddings = embeddings[hardest_neg_dists[1]]

    return hardest_pos_embeddings, hardest_neg_embeddings
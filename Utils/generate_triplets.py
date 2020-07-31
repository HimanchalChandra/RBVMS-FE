import numpy as np
from .utils import extract_face, make_dict
import pickle
import os
import cv2

def generate_triplets(data_dir, triplets_number):
    if os.path.isfile('./data/face_dict.pkl'):
        pkl = open('./data/face_dict.pkl', 'rb')
        face_dict = pickle.load(pkl)
        pkl.close()
    else:
        face_dict = make_dict(data_dir)
    triplets = []
    if not os.path.isfile('./data/training_triplets_'+ str(triplets_number) +'.npy'):
    	for _ in range(triplets_number):
    	    anc_n_pos_class = np.random.choice(list(face_dict.keys()))
    	    neg_class = np.random.choice(list(face_dict.keys()))

    	    while(len(face_dict[anc_n_pos_class]) < 2):
    	    	anc_n_pos_class = np.random.choice(list(face_dict.keys()))

    	    while(neg_class == anc_n_pos_class):
    	    	neg_class = np.random.choice(list(face_dict.keys()))
    		    
    	    pos_image = np.random.choice(face_dict[anc_n_pos_class])
    	    anc_image = np.random.choice(face_dict[anc_n_pos_class])
    		
    	    while(anc_image == pos_image):
    	    	anc_image = np.random.choice(face_dict[anc_n_pos_class])
    		
    	    neg_image = np.random.choice(face_dict[neg_class])
    	    anc_image_path = anc_n_pos_class + '/' + anc_image
    	    pos_image_path = anc_n_pos_class + '/' + pos_image
    	    neg_image_path = neg_class + '/' + neg_image

    	    anc_img = cv2.imread(data_dir + '/' + anc_image_path)
    	    pos_img = cv2.imread(data_dir + '/' + pos_image_path)
    	    neg_img = cv2.imread(data_dir + '/' + neg_image_path)

    	    print(data_dir + '/' + anc_image_path)
    	    anc_face_num, _ = extract_face(cv2.cvtColor(anc_img, cv2.COLOR_BGR2RGB))
    	    print(data_dir + '/' + pos_image_path)
    	    pos_face_num, _ = extract_face(cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB))
    	    print(data_dir + '/' + neg_image_path)
    	    neg_face_num, _ = extract_face(cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB))
    		
    	    if (anc_face_num == 1 and pos_face_num == 1 and neg_face_num == 1):
    	    	triplets.append([anc_n_pos_class, neg_class, anc_image, pos_image, neg_image, anc_image_path, pos_image_path, neg_image_path])
    		
    	np.save('./data/training_triplets_{}.npy'.format(triplets_number), triplets)
    	triplets_filename = 'training_triplets_' + str(triplets_number) + '.npy'
    else:
    	print('File Found')
    	triplets = np.load('./data/training_triplets_' + str(triplets_number) + '.npy')
    	triplets_filename = 'training_triplets_' + str(triplets_number) + '.npy'
    return triplets, triplets_filename

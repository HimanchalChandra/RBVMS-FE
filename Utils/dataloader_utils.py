import os
from tqdm import tqdm
import pickle
from mtcnn import MTCNN 
import cv2


detector = MTCNN()


def extract_face(image):

    face = detector.detect_faces(image)
    faces_num = len(face)
    box = face[0]['box']        
    face_crop = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized_face_crop = cv2.resize(face_crop, (220, 220))

    return resized_face_crop


def extract_all_faces(image):
    faces = detector.detect_faces(image)
    thickness = 2
    color = (0, 255, 0)
    for face in faces:
        box = face['box']
        start_pt = (box[0], box[1] + box[3])
        end_pt = (box[0] + box[2], box[1])
        image = cv2.rectangle(image, start_pt, end_pt, color, thickness)

    return image


def check_img(img_path):

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = detector.detect_faces(image)
    faces_num = len(face)
    print(faces_num)
    check = False
    if faces_num == 1:
        box = face[0]['box']
        if (box[0] > 0) and (box[1] > 0) and (box[2] > 0) and (box[3] > 0):
            check = True
    
    return check


def make_dict(data_dir, face_dict_dir):
    
    classes = os.listdir(data_dir)
    face_dict = dict()
    for folder in tqdm(classes):
        face_dict[str(folder)] = []
        samples = os.listdir(data_dir + '/' + str(folder))
        for sample in samples:
            sample_path = data_dir + '/' + str(folder) + '/' + str(sample)
            if check_img(sample_path):                
                face_dict[str(folder)].append(sample) 
    f = open(face_dict_dir, "wb")
    pickle.dump(face_dict, f)
    f.close()
    
    return face_dict
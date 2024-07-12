from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo
import cv2
import numpy as np

def glcm(image_path):
    data = cv2.imread(image_path, 0)
    co_matrix = graycomatrix(data, [1], [np.pi/4], None, symmetric=False, normed=False)
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    features = [np.float32(dissimilarity), np.float32(cont), np.float32(corr), np.float32(ener), np.float32(asm), np.float32(homo)]
    print(f"GLCM features shape: {len(features)}")
    return features

def bitdesc(image_path):
    data = cv2.imread(image_path, 0)
    features = bio_taxo(data)
    features = [np.float32(feature) for feature in features]
    required_length = 14  
    if len(features) < required_length:
        features += [np.float32(0)] * (required_length - len(features))
    print(f"BIT features shape (after padding): {len(features)}")
    return features

import numpy as np
from scipy.spatial import distance

def manhattan(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    print(f"Manhattan - v1 shape: {v1.shape}, v2 shape: {v2.shape}")
    if v1.shape != v2.shape:
        raise ValueError(f"Shapes of v1 and v2 do not match: {v1.shape} vs {v2.shape}")
    dist = np.sum(np.abs(v1 - v2))
    return dist

def euclidean(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    print(f"Euclidean - v1 shape: {v1.shape}, v2 shape: {v2.shape}")
    if v1.shape != v2.shape:
        raise ValueError(f"Shapes of v1 and v2 do not match: {v1.shape} vs {v2.shape}")
    dist = np.sqrt(np.sum(v1 - v2) ** 2)
    return dist
    
def chebyshev(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    print(f"Chebyshev - v1 shape: {v1.shape}, v2 shape: {v2.shape}")
    if v1.shape != v2.shape:
        raise ValueError(f"Shapes of v1 and v2 do not match: {v1.shape} vs {v2.shape}")
    dist = np.max(np.abs(v1 - v2))
    return dist

def canberra(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    print(f"Canberra - v1 shape: {v1.shape}, v2 shape: {v2.shape}")
    if v1.shape != v2.shape:
        raise ValueError(f"Shapes of v1 and v2 do not match: {v1.shape} vs {v2.shape}")
    return distance.canberra(v1, v2)

def retrieve_similar_image(features_db, query_features, distance, num_results):
    distances = []
    query_features = np.array(query_features).astype('float')
    query_length = query_features.shape[0]

    for instance in features_db:
        features, label, img_path = instance[: -2], instance[-2], instance[-1]
        features = np.array(features).astype('float')
        features_length = features.shape[0]

        if query_length != features_length:
            print(f"Shapes do not match: query_features shape: {query_features.shape}, features shape: {features.shape}")
            continue  
        
        if distance == 'manhattan':
            dist = manhattan(query_features, features)
        elif distance == 'euclidean':
            dist = euclidean(query_features, features)
        elif distance == 'chebyshev':
            dist = chebyshev(query_features, features)
        elif distance == 'canberra':
            dist = canberra(query_features, features)
        else:
            dist = None
        
        if dist is not None:
            distances.append((img_path, dist, label))
    
    distances.sort(key=lambda x: x[1])
    return distances[: num_results]

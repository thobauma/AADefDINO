import numpy as np
import matplotlib.pyplot as plt

def get_random_classes(number_of_classes: int = 50, min_rand_class: int = 1, max_rand_class: int = 1001, seed: int = 42):
    np.random.seed(seed)
    return np.random.randint(low=min_rand_class, high=max_rand_class, size=(number_of_classes,))

def get_random_indexes(number_of_images: int = 50000, n_samples: int=1000, seed: int = 42):
    np.random.seed(seed)
    return np.random.choice(50000, n_samples, replace=False)

def imshow(img):
    npimg = img.cpu().numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.axis('off')
    plt.show()
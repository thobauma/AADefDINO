import numpy as np

def get_random_classes(number_of_classes: int = 50, min_rand_class: int = 0, max_rand_class: int = 999, seed = 42):
    np.random.seed(seed)
    return np.random.randint(low=min_rand_class, high=max_rand_class, size=(number_of_classes,))

def get_random_indexes(number_of_images: int = 50000, n_samples=1000, seed = 42):
    np.random.seed(seed)
    return np.random.choice(50000, n_samples, replace=False)

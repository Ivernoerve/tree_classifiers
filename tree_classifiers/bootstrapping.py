import numpy as np 

def bootstrapping(data: np.ndarray, labels:np.ndarray = None, random_seed: int = None ) -> np.ndarray:
    '''
    Function to bootstrapp sample from a dataset creating a new 
    dataset by sampling from the original dataset with replacement 
    creating a new dataset with exsisting samples

    data: The dataset to perform bootstrapping on shape features , samples
    labels: The labels with corresponding indexes to the data
    random_seed: Optional argument to give a random seed for reproducability
    ---
    returns a bootsrapped array of the same size as the original dataset. 
    if given labels, it will return a tuple of bootstrapped data and labels 
    '''

    if random_seed is not None:
        random_generator = np.random.default_rng(seed=random_seed)
    else: 
        random_generator = np.random.default_rng()
    
    if labels is None:
        return random_generator.choice(data, data.shape[1], axis = 1)
    
    bootstrapped_idx = random_generator.choice(np.arange(data.shape[1]), data.shape[1])

    bootstrapped_data = data[:, bootstrapped_idx]
    bootstrapped_labels = labels[bootstrapped_idx]
    return bootstrapped_data, bootstrapped_labels


if __name__ == '__main__':
    pass
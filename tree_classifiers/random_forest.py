import numpy as np 
import matplotlib.pyplot as plt 

from decision_tree import Decision_tree
from bootstrapping import bootstrapping

class Random_forest():
    def __init__(self,
                data: np.ndarray,
                labels: np.ndarray,
                n_trees: int, 
                n_features_per_tree: int = None,
                maxdepth: int = 5,
                min_purity = .95
                )  -> None: 
        '''
        Random forest classifier used to classify 
        with an ensemble of grown trees. 

        :param data: Data to train the classifier with shape 
                    features, samples 
        :param labels: Corresponding labels to the trianing data 
        :param n_trees: Number of trees in the ensemble 
        (should be odd numbered)
        :param n_features_per_tree: Number of subset features to 
        to give each tree. 
        :param maxdepth: maximum depth for the trees to train 
        defaults to 5 
        '''
        self.data = data 
        self.labels = labels 
        self.n_trees = n_trees
        self.maxdepth = maxdepth
        self.min_purity = min_purity

        if n_features_per_tree is None:
            self.n_features_per_tree = int(np.sqrt(data.shape[0]))
        else:
            self.n_features_per_tree = n_features_per_tree
        
        
    def _create_tree(self, random_seed: int = None):
        '''
        Private method to create a tree in the ensemble

        :param random_seed: gives the ability to set a seed such 
            that the algorithm is predictable.

        :returns: a trained Decission tree classifier to be added
            to the ensemble  
        '''
        tree_feature_indexes = np.random.choice(np.arange(self.data.shape[0]),
                                                self.n_features_per_tree,
                                                replace=False)


        bootsrapped_data, bootstrapped_labels = bootstrapping(self.data, self.labels)

        feature_bootstrapped_data = bootsrapped_data[tree_feature_indexes]

        tree = Decision_tree(feature_bootstrapped_data,
                            bootstrapped_labels,
                            features = tree_feature_indexes,
                            maxdepth= self.maxdepth,
                            min_purity = self.min_purity)
        
        tree.train()
        return tree


    def train(self):
        self.trees = np.array([self._create_tree() for _ in range(self.n_trees)])

        return self 

    
    def classify(self, data: np.ndarray) -> np.ndarray:
        '''
        method to classify new data with the trained random forest 
        classifier 

        :param data: new data to perform classification on. 

        :returns: a numpy array with labels for the data with corresponding indexes
        '''
        
        classified_labels = np.empty((self.n_trees, data.shape[1]), dtype = int)

        for i, tree in enumerate(self.trees):
            tree_labels = tree.classify(data[tree.features])

            classified_labels[i] = tree_labels

        clasiffication_func = lambda data: np.bincount(data).argmax()

        predicted_labels = np.array(list(map(clasiffication_func, classified_labels.T)))
        


        return predicted_labels



if __name__ == '__main__':
    pass
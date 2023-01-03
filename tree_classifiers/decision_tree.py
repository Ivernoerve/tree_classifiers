import numpy as np
import matplotlib.pyplot as plt

from utilities import print_tree_structure

class Decision_tree():
    def __init__(self,
                data: np.ndarray,
                labels: np.ndarray,
                features: np.ndarray = None,
                maxdepth: int = 0,
                min_purity: float = 0.95) -> None:
        """
        Class implementation of a decision tree ML algorithm
        data: training data for the tree to learn shape (features , samples)
        labels: labels corresponding to the data given shape (0, samples)
        features: features to be used in the classfication
        maxdepth: maximum depth of the tree, if 0 the maxdepth is infinite
        min_purity: minimum accepted purity of a leaf node. 
        """
        self.data = data 
        self.labels = labels 
        
        self.features = features
        self.maxdepth = maxdepth
        self.min_purity = min_purity

        
        self.depth = 0
        # 0 root, 1 branch, 2 leaf
        self.type = 0
        self.children = None

        
        
    def _calculate_entropy(self, labels: np.ndarray):
        _, counts = np.unique(labels, return_counts=True)

        probabilities = counts / len(labels)
        entropy = - probabilities @ np.log2(probabilities)
        
        return entropy

    def _find_best_split(self):
        
        best_entropy = 10
        for i, test_boundaries in enumerate(np.sort(self.data, axis = 1)):
            
            for separation_threshold in test_boundaries:
                feature_data = self.data[i]
                
                lower_ind =  np.where(feature_data <= separation_threshold)
                higher_ind = np.where(feature_data > separation_threshold)

                lower_labels = self.labels[lower_ind]
                higher_labels = self.labels[higher_ind]

                potential_entropy = self._calculate_entropy(lower_labels) + self._calculate_entropy(higher_labels)
                
                if potential_entropy < best_entropy and len(lower_labels != 0) and len(higher_labels) != 0:
                    best_entropy = potential_entropy
                    
                    self.lower_ind = lower_ind
                    self.higher_ind  = higher_ind
                    self.sep_dim = i
                    self.sep_value = separation_threshold
                
                if best_entropy == 0:
                    break
            else: 
                continue
            break
        return self


    def _calculate_purity(self):
        """
        Method to calculate the purity of the nodes 
        ---
        returns: max purity and max label information
        """
        labels, counts = np.unique(self.labels, return_counts=True)

        purity = counts / self.data.shape[1]

        max_purity_index = np.argmax(purity)
        max_purity = purity[max_purity_index]
        max_label = labels[max_purity_index]

        return max_purity, max_label

    def train(self):
        '''
        Method to train the decision tree via a recursion
        algorighm.
        '''

        self._find_best_split()

        max_purity, max_label = self._calculate_purity()
        

        if ((self.depth + 1 <= self.maxdepth or self.maxdepth == 0) and 
        max_purity < self.min_purity):# and 
        #self.IG > 0):

            self.children = []
            
            for sep_indexes in [self.lower_ind, self.higher_ind]:

                child = Decision_tree(self.data[:,sep_indexes[0]],
                                    self.labels[sep_indexes],
                                    self.features,
                                    self.maxdepth,
                                    self.min_purity)
                child.depth = self.depth + 1
                child.type = 2
                if self.type == 2:
                    self.type = 1
                self.children.append(child)
                
                child.train()
               

        else: 
            self.purity = max_purity
            self.node_label = max_label




        return self

        
    def _classify_sample(self, sample):
        if self.type != 2:
            if sample[self.sep_dim] <= self.sep_value:
                return self.children[0]._classify_sample(sample)
            else:
                return self.children[1]._classify_sample(sample)
        else:
            return self.node_label

    def classify(self, data: np.ndarray) -> np.ndarray:
        '''
        Method to classify points with a trained decision tree
        data: data to classify, shape features x samples. 
       ¨ ---
        returns array of classified lables where indexes corresponds 
        to the given sample. 
        '''
        labels = np.array(list(map(self._classify_sample, data.T)), dtype = int)

        return labels 




if __name__ == "__main__":
    pass
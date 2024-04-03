import numpy as np
from ai_assignments.datastructures.queue import Queue
from ai_assignments.decision_tree.dt_node import DecisionTreeNodeBase
from scipy.stats import entropy as sci_entropy

def entropy(y):
    '''Takes in list y and computes its entropy.'''
    positive = sum(y) / len(y)
    negative = 1 - positive
    return sci_entropy([positive, negative], base=2)

def unique_unsorted(L):
    '''Takes in list L and returns its unique values, keeping their order.'''
    _, indeces = np.unique(L, return_index=True)
    return [L[index] for index in sorted(indeces)]

class DecisionTreeNode(DecisionTreeNodeBase):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

        if self.features is not None and self.labels is not None:
            self.split()

    def get_all_possible_split_points(self):
        nr_samples, nr_features = self.features.shape
        split_points = []
        for f_idx in range(nr_features):
            idx_sort = self.features[:, f_idx].argsort()
            self.features = self.features[idx_sort, :] # it makes a difference if features is overwritten or not (sorted differently)
            self.labels = self.labels[idx_sort]

            # TODO: check for consecutive samples whether the labels and features are different
            # be careful to not compare the 0th sample with the last sample when indexing
            # compute splitting values and add to list: split_points.append((f_idx, split_at))
            
            #print(f"Features: {self.features}")
            #print(f"Labels: {self.labels}")
            sorted_indices = self.features[:, f_idx].argsort()
            features_sorted = self.features[sorted_indices, :]
            labels_sorted = self.labels[sorted_indices]

            for i in range(1, nr_samples):
                if labels_sorted[i] != labels_sorted[i - 1]:
                    split_at = (features_sorted[i, f_idx] + features_sorted[i - 1, f_idx]) / 2
                    split_points.append((f_idx, split_at))
        
        return split_points

    def get_optimal_split_point(self):
        split_feature, split_point = None, None

        # TODO: 
        # 1. get all possible split points (self.get_all_possible_split_points())
        # 2. loop over all possible split points that you computed and return the best one
        all_split_points = self.get_all_possible_split_points()
        b_gain = -1
        for f_idx, splt_pnt in all_split_points:
            gain = self.compute_information_gain(self.features[:, f_idx], splt_pnt)
            if gain > b_gain:
                b_gain = gain
                split_feature, split_point = f_idx, splt_pnt

        return split_feature, split_point

    def compute_information_gain(self, x, split_point):
        '''computes the information gain for a given feature x and the given split_point'''

        labels_left_branch = self.labels[x <= split_point]
        labels_right_branch = self.labels[x > split_point]

        weight_left = len(labels_left_branch) / len(x)
        weight_right = len(labels_right_branch) / len(x)

        e_data = entropy(self.labels)
        e_right = entropy(labels_right_branch)
        e_left = entropy(labels_left_branch)
        return e_data - (weight_left * e_left + weight_right * e_right)

    def split(self):
        # TODO: implement the ID3 algorithm based on the lecture slides "The ID3 Algorithm"
        # 1. If all examples X in the current node belong to the same class,
        #    make current node a leaf, label with class (use unique_unsorted)

        # 2. use get_optimal_split_point() to determine best feature for splitting

        # 3. + 4. Create a branch + successor node N_i for each value of the selected feature 
        #         containing a subset of X 
        #         - X <= split_point goes into left_child
        #         - X > split_point goes into right_child
        
        # 5. For each subnode split is called recursively

        if len(unique_unsorted(self.labels)) == 1:
            self.label = self.labels[0]
            
            return

        split_feature, split_point = self.get_optimal_split_point()
    
        left_indices = self.features[:, split_feature] <= split_point
        right_indices = ~left_indices

        left_features = self.features[left_indices, :] 
        left_labels = self.labels[left_indices]

        right_features = self.features[right_indices, :]
        right_labels = self.labels[right_indices]

        self.left_child = DecisionTreeNode(left_features, left_labels)
        self.right_child = DecisionTreeNode(right_features, right_labels)

        self.split_feature = split_feature
        self.split_point = split_point
    # def __str__(self):
    #     if self.split_feature is None or self.split_point is None:
    #         return "split_feature or split_point is none"
    #     else:
    #         left_child_str = str(self.left_child) if self.left_child else "None"
    #         right_child_str = str(self.right_child) if self.right_child else "None"
    #         return f"split_feature: {self.split_feature} split_point: {self.split_point:.2f} | left_child_str: {left_child_str}, right_child_str: {right_child_str}"
class ID3():
    def __init__(self):
        self.root = None

    def fit(self, X, y):
        self.root = DecisionTreeNode(X, y)
        return self

    def __str__(self):
        return str(self.root)
    
    def get_height(self, node):
        if node is None:
            return 0
        return max(self.get_height(node.left_child), self.get_height(node.right_child)) + 1
    
    def print_decision_tree(self):
        height = self.get_height(self.root)
        visited = set()
        frontier = Queue()

        lines = ['']

        previous_level = 1
        frontier.put((self.root, 1))

        while frontier.has_elements():
            current, level = frontier.get()
            if level > previous_level:
                lines.append('')
                previous_level = level
            lines[-1] += current.print_node(height, level)
            if current not in visited:
                visited.add(current)
                if current.left_child is not None:
                    frontier.put((current.left_child, level + 1))
                else:
                    if level < height: frontier.put((DecisionTreeNode(None, None), level + 1))
                if current.right_child is not None:
                    frontier.put((current.right_child, level + 1))
                else:
                    if level < height: frontier.put((DecisionTreeNode(None, None), level + 1))

        for line in lines:
            print(line)
        return None

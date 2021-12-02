from __future__ import division, print_function
import numpy as np
from numpy.lib.arraysetops import unique
#from utils import divide_on_feature, train_test_split, calculate_entropy
from sklearn import metrics
from sklearn import datasets
from data_operation import calculate_entropy
from data_manipulation import divide_on_feature, train_test_split
class DecisionNode():
    # Class representing a decision tree node or leaf in the decision tree

    def __init__(self, feature_i = None, threshold = None, 
                value = None, true_branch = None, false_branch = None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for the feature
        self.value = value                  # Value if the node is a leaf
        self.true_branch = true_branch             # left child branch
        self.false_branch = false_branch    # right child branch

class DecisionTree(object):
    """Parameters:
    min_samples_split: int
        the minimum number of the samples need to make a split when building a tree
    min_impurity: float
        the minimumimpurity required to split the tree
    max_depth: int
        the max depth of a tree
    loss: function
        Loss function that is used for gradient boosting model to calculate impurity"""
    def __init__(self,min_samples_split = 2,min_impurity=1e-7,
        max_depth = float("inf"), loss =None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None # Different trees have different ways of calculate impurity
        self._leaf_value_calculation = None 
        self.one_dim = None
        self.loss = loss

    def fit(self, X, y, loss = None):
        # Building the decision tree
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X,y)
        self.loss = None

    def _build_tree(self,X,y,current_depth=0):
        """Recursively build out the tree and split X and y on the feature of X which best 
        seperate the data"""
        largest_impurity = 0
        best_criteria = None # Feature index and threshold
        best_sets = None # Subsets of the data

        #check if the expansion of y is needed
        if (len(np.shape(y))) == 1:
            y = np.expand_dims(y, axis = 1)
        # axis = 1 means if the dimension of y is 1, add a column to it
        # i.e. make it a 8*1 matrix, 8 rows, 1 column
        
        #Add y as the last column of X
        Xy = np.concatenate((X,y),axis=1)

        n_samples, n_features = np.shape(X)
        # X is an array whose row is the samples, and the column is the feature or attribute
        
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity of each attribute
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                # save all rows and pick the i th column of feature
                unique_values = np.unique(feature_values)

                #iterate through all unique values of feature column i and calculate the impurity

                for threshold in unique_values:
                    #Devide X and y depending on if the feature value of X at index feature_i
                    #meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        #Select the y_values of the two sets
                        y1 = Xy1[:,n_features:]
                        y2 = Xy2[:,n_features:]

                        #Calculate the impurity
                        impurity = self._impurity_calculation(y,y1,y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "left_X": Xy1[:, :n_features], #X of left subtree
                                "left_y": Xy1[:, n_features:], #y of left subtree
                                "right_X": Xy2[:, :n_features],#X of right subtree
                                "right_y": Xy2[:, n_features:] #y of right subtree
                            }

        if largest_impurity > self.min_impurity:
            #Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["left_X"], best_sets["left_y"], current_depth+1)
            false_branch = self._build_tree(best_sets["right_X"], best_sets["right_y"], current_depth+1)
            return DecisionNode(feature_i= best_criteria["feature_i"], threshold=best_criteria["threshold"]
            , true_branch=true_branch, false_branch=false_branch)
            
            # We're at leaf ==> determine value

        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value = leaf_value)
                    
    def predict_value(self, x, tree = None):
        """Do a recursive search down the tree and make a prediction of the data sample by 
        the value of the leaf that we end up at"""

        if tree is None:
            tree = self.root
        
        # If we have a value i.e. we are at the leaf, return value as prediction
        if tree.value is not None:
            return tree.value
        
        # choose the feature that we will test
        feature_value = x[tree.feature_i]
        
        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
            elif feature_value == tree.threshold:
                branch = tree.true_branch

            #Test subtree
            return self.predict_value(x, branch)
    
    def predict(self, X):
        """classify samples one by one and return set of labels"""
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """Recursively print out the dicision tree"""
        if not tree:
            tree = self.root

        #when we are at the leaf,print out the label
        if tree.value is not None:
            print(tree.value)

        # Go deeper down the tree
        else:
            # print the test
            print("%s:%s" % (tree.feature_i, tree.threshold))
            # print the True scenario
            print("%sT-->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # print the False scenario
            print("%sF-->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)

    
class ClassificationTree(DecisionTree):
    #inherit from DecisionTree super class
    def _calculate_info_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1-p) * calculate_entropy(y2)

        return info_gain
    
    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            #Count number of occurence of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common
    
    def fit(self, X,y):
        self._impurity_calculation = self._calculate_info_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X ,y)




def main():

    print("____Classification Tree____")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)

    clf = ClassificationTree()
    #Declare object
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    
    
if __name__ == "__main__":
    main()



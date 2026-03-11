import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DTWrapper():

    def __init__(self, dt):
        self.n_estimators   = 1
        self.estimators_    = [dt]
        self.n_features_in_ = dt.n_features_in_


def extract(model):
    # if this is a decision tree, wrap it inside a fake random forest
    if isinstance(model, DecisionTreeClassifier):
        model = DTWrapper(model)

    # each entry has 2*num_feature elements, i.e., each feature has two elements:
    #   [feature_0_left_threshold, feature_0_right_threshold, feature_1_left_threshold, feature_1_right_threshold, ...]
    # each leaf node should have one entry, recording all the features (and left and right threshold) involved in its path (from root to the leaf node)
    # nan means no threshold for that feature
    th_map = np.zeros((1,  2*model.n_features_in_))
    th_map[:] = np.nan

    leaf_value = np.zeros((1, 2))  # to store all this leaf nodes' value
    tree_value = np.zeros((1, 1))  # to store all this leaf nodes' tree ID

    for j in range(model.n_estimators):
        estimator = model.estimators_[j]
        tree = estimator.tree_

        children_left  = tree.children_left               # for each node, the index of its left child
        children_right = tree.children_right              # for each node, the index of its right child
        feature        = tree.feature                     # for each node, the feature index choosen to split (-2 means this is a leaf node)
        threshold      = tree.threshold                   # for each node, the threshold choosen to split (-2 means this is a leaf node)
        value          = tree.value                       # for each node, a list containing the percentage of each class in the node
        leafNode       = np.argwhere(tree.feature == -2)  # index of all leaf nodes

        # go through all leaf nodes
        for i in range(len(leafNode)):
            currentNode = leafNode[i]  # the index of current leaf node

            leaf_value = np.vstack((leaf_value, value[currentNode, 0, :]))  # put this leaf node's value (class percentage) in leaf_value
            tree_value = np.vstack((tree_value, j))                         # put the tree ID of this leaf node in tree_value

            th_map_temp = np.zeros((1, 2*model.n_features_in_))  # temporary entry for this leaf node to add into th_map
            th_map_temp[:] = np.nan

            # traverse back to root
            while currentNode != 0:
                # find parent node whose left child is currentNode
                if (np.argwhere(children_left == currentNode).size != 0):
                    prevNode = np.argwhere(children_left == currentNode)[0][0]  # index of its parent

                    feature_temp   = feature[prevNode]
                    threshold_temp = threshold[prevNode]

                    # update the threshold of the feature
                    if np.isnan(th_map_temp[0, 2*feature_temp+1]):
                        th_map_temp[0, 2*feature_temp+1] = threshold_temp
                    else:
                        th_map_temp[0, 2*feature_temp+1] = min(th_map_temp[0,2*feature_temp+1], threshold_temp)

                    currentNode = prevNode

                # find parent node whose right child is currentNode
                else:
                    prevNode = np.argwhere(children_right == currentNode)[0][0]  # index of its parent

                    feature_temp   = feature[prevNode]
                    threshold_temp = threshold[prevNode]

                    # update the threshold of the feature
                    if np.isnan(th_map_temp[0,2*feature_temp]):
                        th_map_temp[0,2*feature_temp] = threshold_temp
                    else:
                        th_map_temp[0,2*feature_temp] = max(th_map_temp[0,2*feature_temp], threshold_temp)

                    currentNode = prevNode

            th_map = np.vstack((th_map, th_map_temp)) # put this leaf node's information into th_map

    # remove the first entry, which is zero
    th_map     = np.delete(th_map, 0, 0)      # shape [num_leaf_nodes, num_features*2]
    leaf_value = np.delete(leaf_value, 0, 0)  # shape [num_leaf_nodes, 2]
    tree_value = np.delete(tree_value, 0, 0)  # shape [num_leaf_nodes, 1]

    # put them all together
    acam_map = np.hstack((th_map, np.hstack((leaf_value, tree_value))))  # shape [num_leaf_nodes, num_features*2+2+1]

    # only keep the leaf nodes that predicts a 1
    acam_map = acam_map[acam_map[:, model.n_features_in_*2] < acam_map[:, model.n_features_in_*2 + 1]]
    return acam_map[:, :model.n_features_in_*2]

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model


# ============================================================
# DLIM Tree Structure
# ============================================================

TREE_STRUCTURE = {
    9:  (4, 5),
    10: (1, 8),
    11: (0, 2),
    12: (7, 9),
    13: (6, 10),
    14: (12, 13),
    15: (3, 11),
    16: (14, 15)}


LEAF_NODE_NAMES = {
    0: "GAN_FS",
    1: "GAN_ES",
    2: "GAN_EM",
    3: "GAN_AM",
    4: "DIFF_T2I",
    5: "DIFF_I2I",
    6: "DIFF_FS",
    7: "DIFF_FE",
    8: "CAMERA"}


DECISION_NODE_NAMES = {
    9:  "D7",
    10: "D6",
    11: "D5",
    12: "D4",
    13: "D3",
    14: "D1",
    15: "D2",
    16: "ROOT"}


# ============================================================
# Convolution Module
# ============================================================

def conv_module(x, filters, name_prefix,strides=1):

    x = layers.SeparableConv2D(filters=filters,
                               kernel_size=3,
                               strides=strides,   
                               padding="same",
                               use_bias=False,
                               name=f"{name_prefix}_sepconv")(x)

    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

    x = layers.ReLU(name=f"{name_prefix}_relu")(x)

    return x



# ============================================================
# Leaf Node
# Output:
#   q_i : scalar probability/logit
# ============================================================

def build_leaf_node(input_tensor, node_id):

    node_name = LEAF_NODE_NAMES[node_id]

    x = input_tensor
    
    x = conv_module(x,
                    filters=32,
                    name_prefix=f"leaf_{node_id}_{node_name}_block1",
                    strides=2)

    x = layers.AveragePooling2D(pool_size=2,
                                name=f"leaf_{node_id}_{node_name}_pool1" )(x)

    x = conv_module(x,
                    filters=64,
                    name_prefix=f"leaf_{node_id}_{node_name}_block2",
                    strides=2)

    x = layers.AveragePooling2D(pool_size=2,
                                name=f"leaf_{node_id}_{node_name}_pool2")(x)

    x = conv_module(x,
                    filters=128,
                    name_prefix=f"leaf_{node_id}_{node_name}_block3",
                    strides=2)

    x = layers.AveragePooling2D(pool_size=2,
                                name=f"leaf_{node_id}_{node_name}_pool3")(x)

    x = conv_module(x,
                    filters=256,
                    name_prefix=f"leaf_{node_id}_{node_name}_block4",
                    strides=2)
    
    x = layers.AveragePooling2D(pool_size=2,
                                name=f"leaf_{node_id}_{node_name}_pool4")(x)
    
    x = layers.Flatten(name=f"leaf_{node_id}_{node_name}_flatten")(x)
    
    x = layers.Dense(64,
                     activation="relu",
                     name=f"leaf_{node_id}_{node_name}_fc1")(x)
    
    q = layers.Dense(1,
                     activation=None,
                     name=f"leaf_{node_id}_{node_name}_output")(x)
    return q

# ============================================================
# Decision Node
# Output:
#   [p_left, p_right]
# ============================================================
def build_decision_node(input_tensor, node_id):

    node_name = DECISION_NODE_NAMES[node_id]

    x = input_tensor

    x = layers.AveragePooling2D(pool_size=2,
                                name=f"decision_{node_id}_{node_name}_pool1")(x)

    x = layers.AveragePooling2D(pool_size=2,
                                name=f"decision_{node_id}_{node_name}_pool2")(x)

    x = conv_module(x,
                    filters=32,
                    name_prefix=f"decision_{node_id}_{node_name}_block1")

    x = layers.AveragePooling2D(pool_size=2,
                                name=f"decision_{node_id}_{node_name}_pool3")(x)


    x = conv_module(x,
                    filters=64,
                    name_prefix=f"decision_{node_id}_{node_name}_block2")

    x = layers.AveragePooling2D(pool_size=2,
                                name=f"decision_{node_id}_{node_name}_pool4")(x)

    x = conv_module(x,
                    filters=128,
                    name_prefix=f"decision_{node_id}_{node_name}_block3")

    x = layers.AveragePooling2D(pool_size=2,
                                name=f"decision_{node_id}_{node_name}_pool5")(x)
    
    x = layers.Flatten(name=f"decision_{node_id}_{node_name}_flatten")(x)
    
    p = layers.Dense(2,
                     activation="softmax",
                     name=f"decision_{node_id}_{node_name}_output")(x)
    return p



# ============================================================
# Compute All Root-to-Leaf Paths
# ============================================================
def find_path(current_node, target_leaf, tree_dict):

    if current_node == target_leaf:
        return []

    if current_node not in tree_dict:
        return None

    left_child, right_child = tree_dict[current_node]

    # left
    left_path = find_path(left_child, target_leaf, tree_dict)

    if left_path is not None:
        return [(current_node, 0)] + left_path

    # right
    right_path = find_path(right_child, target_leaf, tree_dict)

    if right_path is not None:
        return [(current_node, 1)] + right_path

    return None


# ============================================================
# Build DLIM
# ============================================================
def build_DLIB_model(input_shape=(224, 224, 3)):

    inputs = layers.Input(shape=input_shape,
                          name="DLIM_Input")
    
    # ========================================================
    # Build Leaf Nodes
    # ========================================================
    leaf_outputs = {}
    for leaf_id in range(9):
        q = build_leaf_node(inputs, leaf_id)
        leaf_outputs[leaf_id] = q

    # ========================================================
    # Build Decision Nodes
    # ========================================================
    decision_outputs = {}
    for node_id in range(9, 17):
        p = build_decision_node(inputs, node_id)
        decision_outputs[node_id] = p

    # ========================================================
    # Construct Q Vector
    # Q = [q0, q1, ..., q8]
    # ========================================================
    Q = layers.Concatenate(axis=1,
                           name="Q_Vector"
    )([leaf_outputs[i]
       for i in range(9)])
 
    # ========================================================
    # Construct P Vector
    # ========================================================
    path_probabilities = []
    ROOT_NODE = 16

    for leaf_id in range(9):
        path = find_path(ROOT_NODE, leaf_id, TREE_STRUCTURE)
        prob_list = []

        for node_id, branch_idx in path:
            node_prob = decision_outputs[node_id]
            selected_prob = layers.Lambda(lambda x, idx=branch_idx: x[:, idx:idx+1],
                                          name=f"path_leaf_{leaf_id}_select_node_{node_id}_branch_{branch_idx}"
                                          )(node_prob)
            prob_list.append(selected_prob)

        # multiply all probabilities
        path_prob = prob_list[0]
        for i in range(1, len(prob_list)):
            path_prob = layers.Multiply(name=f"path_leaf_{leaf_id}_multiply_{i}"
                                        )([path_prob, prob_list[i]])
            
        path_probabilities.append(path_prob)

    P = layers.Concatenate(axis=1,
                           name="P_Vector")(path_probabilities)
    
    # ========================================================
    # Model
    # ========================================================
    model = Model(inputs=inputs,
                  outputs=[P,Q,],
                  name="DLIM")
    return model








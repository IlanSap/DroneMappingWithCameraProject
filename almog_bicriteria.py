import math
import matplotlib.pyplot as plt
import numpy as np
# by doing bicritiria we create the corset.
# the node of the division at every step of the bicriteria
class BiNode:
    def __init__(self):
        self.start_index = None
        self.end_index = None
        self.subparts = []
        self.points = []
        self.mse = None


# input: points_list - a list of the points we would like to make into a coreset, and a number k - the size of the division tree
# output: the OPT cost for all of the points
def calc_bicriteria_opt(points_list, k):
    d = len(points_list[0]) - 1  # the dimension of the points # index + value
    cells_list = []  # the list of cells we extracted during the algorithm
    p_list = points_list.copy()  # to avoid override of the original list

    # dividing the data and removing the cell with minimal MSE every iteration
    size = (len(p_list))-2

    node = BiNode()
    while (len(p_list)) >pow(k, d):
        root = divide_bicriteria(p_list, k, d)
        # print_div(p_list, root)
        cell = get_min_bi_cell(root, d)  # getting the cell with the minimal MSE
        node.points.append(cell.points[0])
        cells_list.append(cell)
        # Convert lists to tuples before creating the sets
        p_list = set(tuple(p) for p in p_list)
        cell_points = set(tuple(point) for point in cell.points)

        # Remove the points of the cell we extracted
        p_list -= cell_points
    # calculating OPT, which is the sum of our cells_list's cells MSE
    opt = 0
    for c in cells_list:
        opt += c.mse
    if(len(node.points)>0):
        opt += calc_mse(node, d)

    last_node = BiNode()
    last_node.points = p_list
    opt += calc_mse(last_node,d)
    return opt


# input: points - the points of a cell, weights - the weights of the points, eps - 1/the size of the wanted bicriteria division
# output: a group of size eps * len(points) of the original points
def get_bicriteria_representatives(points, weights, eps):
    d = len(points[0]) - 1
    # transforming the points from tuples to lists
    new_points = [list(p) for p in points]
    # adding the weights as another dimension to the points
    for i in range(len(new_points)):
        new_points[i].append(weights[i])
    # doing the bicriteria division on the weighted points
    tree = divide_bicriteria(new_points, 1/eps, d)
    cells_rep = get_cells_rep(tree, d)
    # splitting the points into a vector of points and a separated vector of weights
    rep_weights = []
    for i in range(len(cells_rep)):
        rep_weights.append(cells_rep[i][d+1])
        del cells_rep[i][-1]
    # transforming back the points from lists to tuples
    cells_rep = [tuple(p) for p in cells_rep]
    return cells_rep, rep_weights


# input: tree - the root of the bicriteria division tree, d - the current dimension
# output: a list of the cell's representatives
def get_cells_rep(tree, d):
    # is the node a leaf?
    if len(tree.points) > 0:
        # getting the node with the maximum weight, and returning it
        max_w_p = tree.points[0]
        for p in tree.points:
            if p[d+1] > max_w_p[d+1]:
                max_w_p = p
        return [max_w_p]
    cells_rep = []
    # recursive call
    for n in tree.subparts:
        cells_rep += get_cells_rep(n, d)
    return cells_rep


# input: points_list - a list of the points we would like to make into a coreset, k - the size of the division tree and d - the dimension of the points
# output: the root of the tree that is representing the bicriteria division
def divide_bicriteria(points_list, k, d):
    root = BiNode()
    points_list = sorted(points_list)
    root.points = points_list
    divide_bicriteria_recursion(root, k, d, d)

    root.start_index = root.subparts[0].start_index
    root.end_index = root.subparts[-1].end_index

    return root


# the recursive part for the bicriteria division tree. dim - the current dimension being handled
def divide_bicriteria_recursion(node, k, d, dim):
    if dim == 0:
        node.mse = calc_mse(node, d)
        return
    points_list = sorted(node.points, key=lambda s: [s[d-dim]])  # sorting the list with the values of the current dimension as the key
    cur_subpart = []
    cur_p_amt = 0  # the amount of points in cur_subpart
    p_left_amt = len(points_list)  # the amount of points left to handle
    cur_k = k
    price = max(p_left_amt / (cur_k), 1)  # updated amount limit for current dimension
    for p in points_list:
        cur_p_amt += 1
        p_left_amt -= 1
        if cur_p_amt > math.floor(price) and cur_k > 1:  # separate the points in cur_subpart and start recursion
            add_bi_node(node, cur_subpart, k, d, dim)
            cur_subpart = [p]
            cur_p_amt = 1
            if cur_k != 1:
                cur_k -= 1
            price = max(p_left_amt / (cur_k), 1)  # updated amount limit for the points left in the current dimension
        else:
            cur_subpart.append(p)
    # check if there are more points and if there are, insert them as well
    if len(cur_subpart) > 0:
        add_bi_node(node, cur_subpart, k, d, dim)
    node.points = []
    return


# input: father_node - a node, p_list - the current list of points (subpart of the entire list), k - as mentioned above, d - as mentioned above, dim - the current dimension
# no output. Adds a new node as a child of father_node and calls the divide_bicriteria_recursion method on it
def add_bi_node(father_node, p_list, k, d, dim):
    next_node = BiNode()
    next_node.start_index = p_list[0][d - dim]
    next_node.end_index = p_list[-1][d - dim]
    next_node.points = p_list
    divide_bicriteria_recursion(next_node, k, d, dim - 1)  # the recursive call!
    father_node.subparts.append(next_node)
    return


# calculates the MSE for the node's points
def calc_mse(node, d):
    p_avg = 0
    mse = 0
    for p in node.points:
        p_avg += p[d]
    p_avg /= len(node.points)
    for p in node.points:
        mse += pow(p[d] - p_avg, 2)
    return mse


# input: root - a node, dim- the current dimension
# output: the leaf of the tree with the lowest MSE
def get_min_bi_cell(root, dim):
    if dim == 0:
        return root
    min_node = None
    for n in root.subparts:
        cur_node = get_min_bi_cell(n, dim-1)
        if min_node is None or cur_node.mse < min_node.mse:
            min_node = cur_node
    return min_node


# FOR DEBUG
# gets the vector of points and a division tree and shows it using matplotlib
def print_div(vec, tree):
    vec = list(vec)
    vec_avg = 0
    for p in vec:
        vec_avg += p[len(p)-1]
    vec_avg = sum(x[len(vec[0])-1] for x in vec) / len(vec)
    vec2 = [x for x in vec if x[len(vec[0])-1] > 0.5]

    vx = [p[0] for p in vec2]
    vy = [p[1] for p in vec2]
    plt.scatter(vx, vy, s=3, marker="o", c="y")

    vec2 = [x for x in vec if x[len(vec[0])-1] < 0.5]

    vx = [p[0] for p in vec2]
    vy = [p[1] for p in vec2]
    plt.scatter(vx, vy, s=3, marker="o", c="c")

    min_start = tree.subparts[0].subparts[0].start_index
    max_end = tree.subparts[0].subparts[-1].end_index

    for sp in tree.subparts:
        min_start = min(min_start, sp.subparts[0].start_index)
        max_end = max(max_end, sp.subparts[-1].end_index)

    plt_avg_cur = tree.subparts[0].start_index
    for i in range(len(tree.subparts)):
        plt_avg_prev = plt_avg_cur
        if i != len(tree.subparts) - 1:
            plt_avg_cur = (tree.subparts[i].end_index + tree.subparts[i + 1].start_index) / 2
            plt.plot([plt_avg_cur, plt_avg_cur], [min_start, max_end], 'k')
        else:
            plt_avg_cur = tree.subparts[i].end_index
        for j in range(len(tree.subparts[i].subparts) - 1):
            plt_avg_row = (tree.subparts[i].subparts[j].end_index + tree.subparts[i].subparts[j + 1].start_index) / 2
            plt.plot([plt_avg_prev, plt_avg_cur], [plt_avg_row, plt_avg_row], 'k')

    plt.show()

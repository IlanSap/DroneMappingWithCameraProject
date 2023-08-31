import random
from io import StringIO

import numpy as np
from sklearn.datasets import load_iris

import alaaboost
import almog_bicriteria as bcr
import almog_coreset_to_points as ctp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

global D
#D = 19 # the size of dimensions, the size of the vectors - 1 (since at the last index we have the value)
D=1
DECIMAL_PRECISION = 2  # the precision of the decimal numbers
IMG_NAME = "testing3.png"


# change the decimal precision of x (the number of digits after the '.')
def dec_prec(x):
    dec_str = "{0:." + str(DECIMAL_PRECISION) + "f}"
    return float(dec_str.format(x))


# the class of a node in the coreset tree
class Node:
    def __init__(self):
        self.parent = None
        self.start_index = None
        self.end_index = None
        self.subparts = []  # will be empty for every node that is a leaf
        self.points = []  # will be empty for every node that isn't a leaf
        self.coreset = None  # will be 'None' for every node that isn't a leaf


# the info of the coreset cells
class Coreset:
    def __init__(self):
        self.cara_points = None
        self.cara_weights = None
        self.cara_idx = None
        self.average = None


# input: points_list - the list of points to divide into coresets, eps - 1/the amount of rows in every dimension, T - the price value (epsilon^D * OPT)
# output: a node that is the root of the coreset division tree
def divide_into_coresets(points_list, eps, T):
    if not check_valid_input_for_coreset_multi_dim(points_list, eps,D):
        # input isn't valid
        return
    root = Node()
    points_list.sort()
    root.points = points_list
    # the recursive call
    divide_into_coresets_recursion(root, eps, T)
    root.start_index = root.subparts[0].start_index
    root.end_index = root.subparts[-1].end_index
    # updating the Caratheodory points and average values
    update_coreset_data(root)
    return root


# the recursive part of divide_into_coresets. Computes the division at a specific dimension and then recursively handles the next one.
def divide_into_coresets_recursion(node, eps, T, dim= D):

    if dim == 0:
        return
    # Using the "sorted" function here is very efficient since it is using the "Timsort" algorithm, which finds
    # subsets that are already sorted and merges them. We have a max of 3 sorted subsets (so we use the "sorted"
    # function).
    points_list = sorted(node.points, key=lambda k: [k[D - dim]])  # sort the points by the values of the current dimension
    cur_subpart = []  # the points we will separate once their MSE reaches the price amount
    price = T / pow(eps, dim - 1)  # updated variance limit for current dimension
    mse_parts = [0, 0, 0]  # a list containing: number of points, sum of points, sum of (points)^2

    # we are trying to insert one row of points at a time, so we will use the next lists and variable to handle one row at a time
    try_subpart = []  # the points currently in our row
    try_mse_parts = [0, 0, 0]  # the same list as mse_parts, but only for the current row
    cur_dim_value = points_list[0][D-dim]  # the value of the current row (e.g. "5" to represent the row of points with the value 5)

    for p in points_list:
        if calc_mse(mse_parts, p[D]) > price:
            # we exceed the price with the new point, so separate the current points and create a new node
            if len(cur_subpart) > 0:  # separate the points currently in cur_subpart and start recursion
                add_node(node, cur_subpart, eps, T, dim)
                mse_parts = try_mse_parts
                cur_subpart = []
            else:
                # "cut" the row here
                add_node(node, try_subpart, eps, T, dim)
                mse_parts = [0, 0, 0]
                try_mse_parts = [0, 0, 0]
                try_subpart = []
        # add the point
        if p[D - dim] != cur_dim_value:
            # new row, insert try_subpart into cur_subpart
            cur_subpart += try_subpart[:]
            try_subpart = []
            try_mse_parts = [0, 0, 0]
        # inserting the point p into the current row and updating mse_parts, try_mse_parts and cur_dim_value
        try_subpart.append(p)
        mse_parts[0] += 1 #counts
        mse_parts[1] += p[D] #sum of points
        mse_parts[2] += pow(p[D], 2) # sum in power of 2
        try_mse_parts[0] += 1
        try_mse_parts[1] += p[D]
        try_mse_parts[2] += pow(p[D], 2)
        cur_dim_value = p[D - dim]

    # make a node from the points that are left
    if len(cur_subpart) > 0 or len(try_subpart) > 0:
        cur_subpart += try_subpart[:]
        add_node(node, cur_subpart, eps, T, dim)

    node.points = []  # the points have been separated at this dimension, we can delete them from this dimension to avoid duplicates
    return


# input: father_node - the node that will be the parent of the new node, p_list - the list of points in the new node, eps, T, dim
# no output, creates a new node and calls the recursive call on it
def add_node(father_node, p_list, eps, T, dim):

    next_node = Node()
    # the edges at this dimension
    next_node.start_index = p_list[0][D - dim]
    next_node.end_index = p_list[-1][D - dim]
    next_node.points = p_list
    next_node.parent = father_node
    divide_into_coresets_recursion(next_node, eps, T, dim - 1)  # the recursive call
    father_node.subparts.append(next_node)
    return


# calculates the new variance using the old variance's data and the new value
def calc_mse(v_list, p_val):
    #p_sum_pow = 0
    p_num = v_list[0] + 1
    p_sum = v_list[1] + p_val
    #p_sum = v_list[1] + sum(p_val)
    #for pv in p_val:
        #p_sum_pow = p_sum_pow + v_list[2] + pow(pv, 2)
    p_sum_pow = v_list[2] + pow(p_val, 2)
    new_avg = p_sum / p_num
    new_mse = (p_sum_pow - (2 * p_sum * new_avg) + p_num * pow(new_avg, 2))
    return new_mse


# a function that for each leaf (which represents a cell) will calculate the coreset values (caratheodory, average, etc)
def update_coreset_data(root):
    if len(root.points) > 0:
        root.coreset = Coreset()
        # empty np-arrays to insert the Caratheodory data into
        cara_p = np.zeros((len(root.points), 3))
        cara_w = np.ones((len(root.points), 1))
        p_sum = 0  # to compute average
        for i, p in enumerate(root.points):
            p_sum += p[D]
            cara_p[i][0] = p[D]
            cara_p[ i][1] = 1
            cara_p[i][2] = pow(p[D], 2)
            i += 1
        root.coreset.average = p_sum / len(root.points)  # changing decimal precision
        # using the Caratheodory points computation
        root.coreset.cara_points, root.coreset.cara_weights, root.coreset.cara_idx = alaaboost.updated_cara(cara_p, cara_w, 4) # caratheodory
        # removing "ghost" (0 weight out of index range) points
        is_ghost = False
        real_size = 0
        for i in range(len(root.coreset.cara_idx)):
            if root.coreset.cara_idx[i] >= len(root.points):
                # a ghost point
                is_ghost = True
            else:
                real_size += 1
        # there are ghost points to remove
        if is_ghost:
            new_cara_p = np.zeros((real_size, 3))
            new_cara_w = np.ones((real_size, 1))
            new_idx = np.zeros(real_size).reshape(-1).astype(int)
            for i in range(real_size):
                new_cara_p[i] = root.coreset.cara_points[i]
                new_cara_w[i] = root.coreset.cara_weights[i]
                new_idx[i] = root.coreset.cara_idx[i]
            root.coreset.cara_points = new_cara_p
            root.coreset.cara_weights = new_cara_w
            root.coreset.cara_idx = new_idx
        return
    for node in root.subparts:
        update_coreset_data(node)  # updating the data recursively
    return


# input: the root of the coreset division tree
# output: 2 lists with the Caratheodory points and weights
def get_tree_coreset(tree):
    if len(tree.points) > 0:
        node_csp = []
        node_csw = []
        for i in tree.coreset.cara_idx:
            if len(tree.points) > i:
                node_csp.append(tree.points[i])
        for i in tree.coreset.cara_weights:
            node_csw.append(i)
        return node_csp, node_csw
    if len(tree.subparts) > 0:
        # recurse for the nodes in subparts
        node_csp = []
        node_csw = []
        for s in tree.subparts:
            node_cspn, node_cswn = get_tree_coreset(s)
            node_csp += node_cspn
            node_csw += node_cswn
        return node_csp, node_csw


# input: points_list - a list of points, eps - the epsilon (1/size of row at each dimension)
# output: a boolean stating whether the inputs are valid for the algorithm or not. Prints an error message if False.
def check_valid_input_for_coreset_multi_dim(points_list, eps,D):
    if len(points_list) < 1:
        print("ERROR: List of points is empty")
        return False
    if eps <= 0 or eps > 1:
        print("ERROR: Epsilon must be between greater then 0 and lesser or equal to 1")
        return False
    for p in points_list:
        if len(p) != D+1:
            print("ERROR: All points must be of size D")
            return False
    return True


# input: p_list - the list of points, root - the root of the coreset tree, eps - the epsilon
# no output; prints any logic error in the result of the algorithm
def sanity_check(p_list, root, eps):
    error_list = []  # the list of errors to print at the end of the function
    points_amount = check_points_amount(root)
    if points_amount != len(p_list):
        error_list.append("ERROR: Amount of points is not right")
    points_exist = check_points_exist(root, p_list[:])
    if len(points_exist) != 0:
        error_list.append("ERROR: Points are missing. \nMissing points: " + str(points_exist))
    is_points_same_dimension = check_points_same_dimension(p_list)
    if not is_points_same_dimension:
        error_list.append("ERROR: All points must be of the same size.")
    dimensions_size = check_dimensions_size(root, eps)
    if not dimensions_size:
        error_list.append("ERROR: One of the dimensions is too big")
    is_idx = check_start_end_idx(root)
    if not is_idx:
        error_list.append("ERROR: Start and end indexes aren't valid (one of the start_idx is bigger than his end_idx)")

    if len(error_list) == 0:
        print("Sanity Check: No errors detected")
    else:
        print("Sanity Check: ~ERRORS DETECTED~")
        for err in error_list:
            print("- ", err)
    return


# checks if all of points are in the coreset tree
def check_points_amount(node):
    if len(node.subparts) == 0:
        return len(node.points)
    p_sum = 0
    for p in node.subparts:
        p_sum = p_sum + check_points_amount(p)
    return p_sum


# checks if all of the points are in the coreset tree
def check_points_exist(node, p_list):
    if len(node.points) > 0:
        for p in node.points:
            p_list.remove(p)
        return p_list
    for p in node.subparts:
        p_list = check_points_exist(p, p_list)
    return p_list


# checks if all of the points are of the same dimension
def check_points_same_dimension(p_list):
    d = len(p_list[0])
    for p in p_list:
        if len(p) != d:
            return False
    return True


# checks if all of the dimensions are smaller than 1/epsilon
def check_dimensions_size(node, eps):
    if len(node.subparts) > 1/eps+1:
        return False
    if len(node.subparts) == 0:
        return True
    is_size = True
    for p in node.subparts:
        is_size = is_size and check_dimensions_size(p, eps)
    return is_size


# checks if all of the end indexes are bigger than the start indexes
def check_start_end_idx(node):
    if node.start_index <= node.end_index:
        is_true = True
    else:
        is_true = False
    for n in node.subparts:
        is_true = is_true and check_start_end_idx(n)
    return is_true


# the main function, that does everything.
def to_coreset(vec, epsilon, k):
    # calculating the optimal OPT value
    #print("start calc bicriteria")
    start = time.time()
    t_opt = bcr.calc_bicriteria_opt(vec, k)
    end = time.time()
    print(t_opt)
    #print("end calc bicriteria")
    # t is the price at every cell  # price means how many points we can have in a cell.
    t = t_opt * pow(epsilon, D)
    # dividing to coresets
    #print("start divid into tree")
    tree = divide_into_coresets(vec.copy(), epsilon, t)   # until we haven't reach the price we add more points into a cell. when we get the price we create a node and add it to the tree.
    #print("end divid into tree")
    n_p_list, n_w_list = leafs_to_coreset(tree, D, epsilon)
    # checking that the values of the coreset are valid and are logical
    #sanity_check(vec.copy(), tree, epsilon)
    # taking the coreset division and turning it to points
    #n_p_list, n_w_list = ctp.coreset_to_points(tree, D, epsilon)
    divs = get_divs(tree)
    del divs[-1]
    return n_p_list, n_w_list, divs, (end-start)


def leafs_to_coreset(node, d, eps):
    p_list, w_list = leafs_to_coreset_rec(node, d, eps)
    print("NUMBER OF OUTPUT POINTS: ", len(p_list))
    # normalize the points
    p_amount = ctp.calc_p_amount(node)
    p_sum = sum(w_list)
    for i in range(len(w_list)):
        w_list[i] = (w_list[i] / p_sum) * p_amount

    return p_list, w_list

def leafs_to_coreset_rec(node, d, eps):
    p_list = []
    w_list = []
    # this is a leaf
    if len(node.points) > 0:
        # check how many cara points are in the node
        cp_amount = 0
        for cp in node.coreset.cara_idx:
            if cp < len(node.points):
                cp_amount += 1
        p_amount = node.coreset.cara_weights.copy()
        p_amount = p_amount.reshape(1,-1)
        w_list=[]
        p_amount = p_amount[0].tolist()

        cara_p_list = []
        for p_idx in node.coreset.cara_idx:

            new_point = list(node.points[p_idx]).copy()
            #new_point[d] = node.points[node.coreset.cara_idx[i]][d]
            cara_p_list.append(new_point)

            cara_p_list, cara_w_list = ctp.get_p_weights(cara_p_list, d)
            cara_w_list = p_amount
            if len(cara_p_list) > 0:
                p_list  = cara_p_list
                w_list = cara_w_list
    # recursion- calling recursive function until we reach the leafs of the tree
    if len(node.subparts) > 0:
        for n in node.subparts:
            new_p_list, new_w_list = leafs_to_coreset_rec(n, d, eps)
            p_list += new_p_list
            w_list += new_w_list

    return p_list, w_list


# getting the dividers of the tree
def get_divs(root):
    if len(root.points) > 0:
        return [root.start_index]
    if len(root.subparts) > 0:
        div_list = []
        for n in root.subparts:
            div_list += get_divs(n)
        return div_list


# the cost function
# input: a list of points, a list of weights (both lists have same size)
# output: the weighted MSE cost of the points
def cost(p_list, w_list):
    w_average = 0
    w_sum = sum(w_list)
    # weighted average computation
    for i, p in enumerate(p_list):
        w_average += p[D] * (w_list[i] / w_sum)
    mse_list = []
    # each point's MSE computation
    for p in p_list:
        mse_list.append(pow(p[D] - w_average, 2))
    tot_cost = 0
    # the total weighted MSE computation
    for i, mse in enumerate(mse_list):
        tot_cost += mse * w_list[i]
    return tot_cost


def decisionTreeOnUniformData():
    #### unifrom random data
    dims = [15]
    build_coreset_time = []
    times_with_corset = []
    times_without_corset = []
    scores = []
    arr = np.random.uniform(0, 1, size=(15 * 32561))
    bin_arr = arr.reshape(1, (15 * 32561))

    D = 15
    vec = []
    min_weight = 0.01
    index = 0
    for i in dims:
        for x in range(32561):
            vars = []
            vars.append(x)
            for j in range(D):
                vars.append(bin_arr[0, index])
                index = (index + 1)
            vars = tuple(vars)
            vec.append((vars))

        start = time.time()
        n_p_list, n_w_list, divs = to_coreset(vec, 0.5, 2)
        end = time.time()
        build_coreset_time.append(end - start)

        not_zeros = [i for i in range(275)]
        X = np.array(n_p_list)[not_zeros, :]
        Y = np.arange(0, X.shape[0], 1)

        x_tr, x_tes, y_tr, y_tes = train_test_split(
            X[:, [ 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14,15]],  # x
            Y,  # y
        )
        print("coreset size", x_tr.shape)
        print(x_tr.shape)
        print(y_tr.shape)
        start2 = time.time()
        clf = tree.DecisionTreeClassifier()
        print("start fit on corset data")
        clf = clf.fit(x_tr, y_tr)
        print("finish fit on coreset data")
        scores.append( clf.score(x_tes,y_tes))
        end = time.time()

        times_with_corset.append(end - start)

        arr = arr.reshape(32561, 15)
        arr = np.array(arr)
        Y = np.arange(0, arr.shape[0], 1)
        x_tr, x_tes, y_tr, y_tes = train_test_split(
            arr[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],  # x
            Y,  # y
            test_size=.1
        )
        start = time.time()
        clf = tree.DecisionTreeClassifier()
        print("start fit on origin data")
        clf = clf.fit(x_tr, y_tr)
        print("finish fit on origin data")
        scores.append(clf.score(x_tes, y_tes))
        end = time.time()
        times_without_corset.append(end - start)


    print("Results on uniform data:")
    print("WITHOUT coreset: ", times_without_corset)
    print("WITH CORESET: ", times_with_corset)
    print("build coreset time: ", build_coreset_time)
    print("score on original data: ", scores[1], "scores on coreset data: ", scores[0])


def readData(path):
    df = pd.read_csv(path)
    columns = list(df.head(0))  # columns names
    for column in columns:
        df[column] = df[column].replace('?', "0")

    df = df.fillna(0)
    D = len(columns)
    print("Dim is: ", D)
    return df, D

def preprocesingAdultData(df):
    #convert nimunal data -> ordinal data: label encoder

    Labelworkclass = LabelEncoder()
    df['workclassLE'] = Labelworkclass.fit_transform(df['workclass'])

    Labeleducation = LabelEncoder()
    df['educationLE'] = Labeleducation.fit_transform(df['education'])

    Labelmarital_status = LabelEncoder()
    df['marital.statusLE'] = Labelmarital_status.fit_transform(df['marital.status'])
    Labeloccupation = LabelEncoder()
    df['occupationLE'] = Labeloccupation.fit_transform(df['occupation'])
    Labelrelationship = LabelEncoder()
    df['relationshipLE'] = Labelrelationship.fit_transform(df['relationship'])
    Labelrace = LabelEncoder()
    df['raceLE'] = Labelrace.fit_transform(df['race'])
    Labelsex = LabelEncoder()
    df['sexLE'] = Labelsex.fit_transform(df['sex'])


    Labelnative_country = LabelEncoder()
    df['native.countryLE'] = Labelnative_country.fit_transform(df['native.country'])
    Labelincome = LabelEncoder()
    df['incomeLE'] = Labelincome.fit_transform(df['income'])

    df = df.drop([ 'workclass', 'education',  'marital.status', 'occupation',
                  'relationship', 'race', 'sex',  'native.country',
                  'income'], axis='columns')

    return  df

def preprocesingBlackFridayData(df):
    df = df.drop(['User_ID','Product_ID'], axis = 1 )

    df = pd.get_dummies(df, prefix=['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'], drop_first=['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])

    """
        LabelGender = LabelEncoder()
        df['GenderLE'] = LabelGender.fit_transform(df['Gender'])
    
        LabelAge = LabelEncoder()
        df['AgeLE'] = LabelAge.fit_transform(df['Age'])
    
        LabelCity_Category = LabelEncoder()
        df['City_CategoryLE'] = LabelCity_Category.fit_transform(df['City_Category'])
    
        LabelStay_In_Current_City_Years = LabelEncoder()
        df['Stay_In_Current_City_YearsLE'] = LabelStay_In_Current_City_Years.fit_transform(df['Stay_In_Current_City_Years'])
    
        df = df.drop(['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'], axis=1)
     """
    purchast = df['Purchase']
    df=df.drop('Purchase', axis=1)
    df['Purchase']= purchast



    cols = list(df.head(0))
    print(cols)
    return df , len(cols)

def classification_origin_data(df):

    X = df.drop('incomeLE', axis=1)
    Y = df['incomeLE']
    # split dataset: training 90% testing 10%
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=.1
    )
    start = time.time()
    model = tree.DecisionTreeClassifier( )
    model = model.fit(x_train, y_train)  # find the best tree with the  min mse

    origin_data_score = (model.score(x_test, y_test))
    end = time.time()
    origin_data_time = (end-start)

    from IPython.display import Image
    import pydotplus
    dot_data = tree.export_graphviz(model,
                                    feature_names=df.columns[0:-1],
                                    class_names=['<=50K', '>50K'],
                                    out_file=None,
                                    filled=True,
                                    rounded=True)

    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())
    graph.write_png('tree.png')
    return x_train, y_train, x_test, y_test, origin_data_time, origin_data_score


def create_vec_from_dataset(df):
    values = np.array(df.values)
    var = []
    vec = []
    rows = len(values)
    # preprocessing the data to fit to_coreset function.
    for i in range(0, rows, 1):
        var.append(i)
        for j in range(D):
            var.append(values[i][j])
        var = tuple(var)
        vec.append(var)
        var = []

    return vec

def classification_coreset(df, D, x_train, y_train, x_test, y_test, origin_data_time, origin_data_score, writer):
    coreset_size = []
    cols = df.columns
    frames = [x_train, y_train]
    data = pd.concat(frames, axis=1)
    vec = create_vec_from_dataset(data)
    epsilon = 0.8
    min_weight = 0.0000000001

    std_times = []
    std_scores = []
    std_coreset_size = []
    std_build_time = []
    for j in range (10):
        building_corset_time_temp = []
        scores_temp = []
        times_temp = []
        coreset_size_temp = []

        for i in range(5):
            print("epsilon=", epsilon)
            start_to_coreset = time.time()
            n_p_list, n_w_list, divs, bicreteria_time = to_coreset(vec, epsilon, 2)
            end_to_coreset = time.time()
            building_coreset_t = end_to_coreset - start_to_coreset
            building_corset_time_temp.append(building_coreset_t)
            not_zeros = [i for i in range(len(n_w_list)) if n_w_list[i] > min_weight]
            X = np.array(n_p_list)[not_zeros, :]
            print(len(not_zeros))
            coreset_size_temp.append(len(not_zeros))
            size = len(not_zeros)
            x_tr, x_tes, y_tr, y_tes = train_test_split(
                X[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14]],  # x
                X[:, [15]],  # y
                test_size=.1
            )
            acc_coreset =0
            start= 0
            end =0
            try:
                start = time.time()
                model2 = tree.DecisionTreeClassifier()
                model2 = model2.fit(x_tr, y_tr.astype('int'))
                end = time.time()
                coreset_weights = np.array(n_w_list)[0: (x_test.shape[0])]
                acc_coreset = model2.score(x_test, y_test)#, sample_weight=coreset_weights)
                scores_temp.append(acc_coreset)
                times_temp.append(end - start)
            except:
                print("memory error", "coreset shape is:" ,len(not_zeros))
            writer.writerow([size,building_coreset_t, acc_coreset , origin_data_score, (end - start) ,origin_data_time, bicreteria_time, epsilon])


        times.append(sum(times_temp)/5)
        building_corset_time.append( sum(building_corset_time_temp)/5)
        scores.append(sum(scores_temp)/5)
        coreset_size.append(sum(coreset_size_temp)/5)

        std_build_time.append(np.std(building_corset_time_temp))
        std_coreset_size.append(np.std(coreset_size_temp))
        std_scores.append(np.std(scores_temp))
        std_times.append(np.std(times_temp))


        epsilon = epsilon-0.025

    return times,std_times, building_corset_time, std_build_time ,scores, std_scores, coreset_size, std_coreset_size

def classification_uniform(df, D, x_train, y_train ,x_test,y_test, coreset_size):
    scores = []
    times = []
    frames = [x_train, y_train]
    df = pd.concat(frames, axis=1)
    std_times = []
    std_scores = []
    scores_all= []
    times_all=[]
    for size in coreset_size:
        scores_temp = []
        times_temp = []

        for i in range(5):
            random_indexes = np.array(np.random.uniform(low=0, high=df.shape[0], size=int(size)), int)
            X = np.array(df.values)[random_indexes, :]
            x_tr, x_tes, y_tr, y_tes = train_test_split(
                X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]],  # x
                X[:, [14]],  # y
                test_size=.1
            )
            try:
                start = time.time()
                model2 = tree.DecisionTreeClassifier()
                model2 = model2.fit(x_tr, y_tr.astype('int'))
                end = time.time()
                scores_temp.append(model2.score(x_test, y_test))
                scores_all.append(model2.score(x_test, y_test))
                times_temp.append(end - start)
                times_all.append(end - start)
            except:
                print("memory error")

        scores.append(sum(scores_temp)/5)
        times.append(sum(times_temp)/5)

        std_scores.append(np.std(scores_temp))
        std_times.append(np.std(times_temp))

    csv_input = pd.read_csv('classi_res.csv')
    csv_input['Uniform_acc'] = scores_all
    csv_input['Uniform_time'] = times_all
    csv_input.to_csv('classi_res.csv', index=False)

    print("times of unifrom data:", times)
    print("std times of uniform data: ", std_times)
    print("scores of unifrom data:", scores)
    print("std scores of uniform data: ", std_scores)

def regration_origin_data(df):
    X = df.drop('Purchase', axis=1)
    Y = df['Purchase']
    # split dataset: training 90% testing 10%
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=.1
    )

    model = tree.DecisionTreeRegressor()
    start = time.time()
    model = model.fit(x_train, y_train)
    end = time.time()
    y_pred = model.predict(x_test)
    y_pred = np.array(y_pred,int)
    origin_data_score = metrics.accuracy_score( y_test,y_pred, normalize=True, sample_weight=None)

    origin_data_time = (end - start)
    df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    RMSE_origin =  np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('RMSE (Root Mean Squared Error) of origin data:', RMSE_origin)

    # 3
    model2 = tree.DecisionTreeRegressor(max_depth=3)
    start = time.time()
    model2.fit(x_train, y_train)
    end = time.time()
    y_pred2 = model2.predict(x_test)

    print("RMSE is ", np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
    print("training time is", (end - start), "for depth =3")

    # 5
    model2 = tree.DecisionTreeRegressor(max_depth=5)
    start = time.time()
    model2.fit(x_train, y_train)
    end = time.time()
    y_pred2 = model2.predict(x_test)
    print("RMSE is ", np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
    print("training time is", (end - start), "for depth =5")

    # 7
    model2 = tree.DecisionTreeRegressor(max_depth=7)
    start = time.time()
    model2.fit(x_train, y_train)
    end = time.time()
    y_pred2 = model2.predict(x_test)
    print("RMSE is ", np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
    print("training time is", (end - start), "for depth =7")
    ########

    #
    param_grid = [3, 4, 5]
    print(param_grid)
    min_rsme = 60000
    min_time = 10
    min_leaf_ = 8
    for min_leaf in param_grid:
        model2 = tree.DecisionTreeRegressor(min_samples_leaf=min_leaf)
        start = time.time()
        model2.fit(x_train, y_train)
        end = time.time()
        y_pred2 = model2.predict(x_test)
        rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
        if (rsme < min_rsme):
            min_rsme = rsme
            min_time = (end - start)
            min_leaf_ = min_leaf
    print("RMSE is ", min_rsme)
    print("training time is", min_time, "for best parameter for min_sample_leaf =", min_leaf_)

    #
    param_grid = [8, 10, 12]
    print(param_grid)
    min_rsme = 60000
    min_time = 10
    min_leaf_ = 8
    for min_leaf in param_grid:
        model2 = tree.DecisionTreeRegressor(min_samples_split=min_leaf)
        start = time.time()
        model2.fit(x_train, y_train)
        end = time.time()
        y_pred2 = model2.predict(x_test)
        rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
        if (rsme < min_rsme):
            min_rsme = rsme
            min_time = (end - start)
            min_leaf_ = min_leaf
    print("RMSE is ", min_rsme)
    print("training time is", min_time, "for best parameter for min_samples_split =", min_leaf_)

    # 7
    param_grid = [7, 10, 12, 15, 20, 25, 50]
    print(param_grid)
    min_rsme = 60000
    min_time = 10
    max_leaf_ = 8
    for max_leaf in param_grid:
        model2 = tree.DecisionTreeRegressor(max_depth=max_leaf)
        start = time.time()
        model2.fit(x_train, y_train)
        end = time.time()
        y_pred2 = model2.predict(x_test)
        rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
        if (rsme < min_rsme):
            min_rsme = rsme
            min_time = (end - start)
            max_leaf_ = max_leaf
    print("RMSE is ", min_rsme)
    print("training time is", min_time, "for best parameter for max_depth  =", max_leaf_)

    return x_train, y_train, x_test, y_test, origin_data_time, origin_data_score, RMSE_origin


def regration_coreset(df,D, x_train,y_train, x_test, y_test, RMSE_origin, origin_data_time, writer):
    coreset_size = []

    frames = [x_train, y_train]
    data = pd.concat(frames, axis=1)
    vec = create_vec_from_dataset(data)
    epsilon = 0.8
    min_weight = 0.0000000001
    std_times = []
    std_scores = []
    std_coreset_size = []
    std_RMSE = []
    std_build_time = []
    for i in range (10):
        building_corset_time_temp = []
        scores_temp = []
        times_temp = []
        coreset_size_temp = []
        RMSE_array_temp = []

        for j in range(5):
            X=[]
            print("epsilon = ",epsilon)
            start_to_coreset = time.time()
            n_p_list, n_w_list, divs, bicreteria_time = to_coreset(vec, epsilon, 4)
            end_to_coreset = time.time()
            building_coreset_t = end_to_coreset - start_to_coreset
            building_corset_time_temp.append(building_coreset_t)

            not_zeros = [i for i in range(len(n_w_list)) if n_w_list[i] > min_weight]
            X = np.array(n_p_list)[not_zeros, :]

            print("coreset size", len(not_zeros))
            size = len(not_zeros)
            coreset_size_temp.append(size)

            x_tr, x_tes, y_tr, y_tes = train_test_split(
                X[:, 1:(D)],  # x
                X[:, [D]],  # y
                test_size=.1,
                random_state=42
            )
            model2 = tree.DecisionTreeRegressor()
            start = time.time()
            model2.fit(x_tr, y_tr)
            end = time.time()
            y_pred2 = model2.predict(x_test)
            times_temp.append(end - start)
            RMSE_coreset = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))

            print("RMSE is" , RMSE_coreset)
            RMSE_array_temp.append(RMSE_coreset)

            coreset_weights = np.array(n_w_list)[0: (x_test.shape[0])]
            scores.append(metrics.accuracy_score(y_test, np.array(y_pred2,int), normalize=True))

            writer.writerow([size,building_coreset_t, RMSE_coreset, RMSE_origin, (end - start) ,origin_data_time, bicreteria_time, epsilon])


            #3
            model2 = tree.DecisionTreeRegressor(max_depth=3)
            start = time.time()
            model2.fit(x_tr,y_tr)
            end = time.time()
            y_pred2 = model2.predict(x_test)
            print("coreset size", len(X))

            print ("RMSE is ", np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            print("training time is", (end-start), "for depth =3")

            #5
            model2 = tree.DecisionTreeRegressor(max_depth=5)
            start = time.time()
            model2.fit(x_tr, y_tr)
            end = time.time()
            y_pred2 = model2.predict(x_test)
            print("RMSE is ", np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            print("training time is", (end - start), "for depth =5")

            #7
            model2 = tree.DecisionTreeRegressor(max_depth=7)
            start = time.time()
            model2.fit(x_tr, y_tr)
            end = time.time()
            y_pred2 = model2.predict(x_test)
            print("RMSE is ", np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            print("training time is", (end - start), "for depth =7")
            ########
            #
            param_grid = [3,4,5]
            print(param_grid)
            min_rsme = 60000
            min_time = 10
            min_leaf_ = 8
            for min_leaf in param_grid:
                model2 = tree.DecisionTreeRegressor(min_samples_leaf= min_leaf)
                start = time.time()
                model2.fit(x_tr, y_tr)
                end = time.time()
                y_pred2 = model2.predict(x_test)
                rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
                if (rsme < min_rsme):
                    min_rsme = rsme
                    min_time = (end - start)
                    min_leaf_ = min_leaf
            print("RMSE is ", min_rsme)
            print("training time is", min_time, "for best parameter for min_sample_leaf =", min_leaf_)

            #
            param_grid = [8,10,12]
            print(param_grid)
            min_rsme = 60000
            min_time = 10
            min_leaf_ = 8
            for min_leaf in param_grid:
                model2 = tree.DecisionTreeRegressor(min_samples_split= min_leaf)
                start = time.time()
                model2.fit(x_tr, y_tr)
                end = time.time()
                y_pred2 = model2.predict(x_test)
                rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
                if (rsme<min_rsme):
                    min_rsme= rsme
                    min_time=(end-start)
                    min_leaf_=min_leaf
            print("RMSE is ", min_rsme)
            print("training time is", min_time, "for best parameter for min_samples_split =", min_leaf_)

            # 7
            param_grid = [7,10,12,15,20,25,50]
            print(param_grid)
            min_rsme = 60000
            min_time = 10
            max_leaf_ = 8
            for max_leaf in param_grid:
                model2 = tree.DecisionTreeRegressor(max_depth= max_leaf)
                start = time.time()
                model2.fit(x_tr, y_tr)
                end = time.time()
                y_pred2 = model2.predict(x_test)
                rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
                if (rsme < min_rsme):
                    min_rsme = rsme
                    min_time = (end - start)
                    max_leaf_ = max_leaf
            print("RMSE is ", min_rsme)
            print("training time is", min_time, "for best parameter for max_depth  =", max_leaf_)

        epsilon = epsilon-0.025

        times.append(sum(times_temp)/5)
        building_corset_time.append( sum(building_corset_time_temp)/5)
        scores.append(sum(scores_temp)/5)
        coreset_size.append(sum(coreset_size_temp)/5)
        RMSE_array.append(sum(RMSE_array_temp)/5)

        std_build_time.append(np.std(building_corset_time_temp))
        std_coreset_size.append(np.std(coreset_size_temp))
        std_scores.append(np.std(scores_temp))
        std_times.append(np.std(times_temp))
        std_RMSE.append(np.std(RMSE_array_temp))


    return times, std_times, building_corset_time, std_build_time, scores, std_scores, coreset_size, std_coreset_size, RMSE_array, std_RMSE

def regration_for_uniform(df,D, x_train, y_train, x_test, y_test, coreset_size):
    frames = [x_train, y_train]
    df = pd.concat(frames, axis=1)
    print("num of rows=" , df.shape[0])
    times_all=[]
    RMSE_all=[]
    std_times = []
    std_scores = []
    std_RMSE = []
    for size in range (len(coreset_size)):
        scores_temp = []
        times_temp = []
        RMSE_array_temp = []

        for i in range(5):
            random_indexes = np.array(np.random.uniform(low=0, high=df.shape[0], size=int(coreset_size[size])),int)
            X = np.array(df.values)[random_indexes, :]

            x_tr, x_tes, y_tr, y_tes = train_test_split(
                X[:, 0:(D-1)],  # x
                X[:, [D-1]],  # y
                test_size=.1,
                random_state=42
            )
            model2 = tree.DecisionTreeRegressor()
            start = time.time()
            model2.fit(x_tr, y_tr)
            end = time.time()
            y_pred2 = model2.predict(x_test)
            RMSE_array_temp.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            RMSE_all.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            times_temp.append(end - start)
            times_all.append(end - start)
            try:
                scores_temp.append(metrics.accuracy_score(y_test, np.array(y_pred2,int), normalize=True, sample_weight=None))
            except:
                print("memory error")

        times.append(sum(times_temp)/5)
        scores.append(sum(scores_temp)/5)
        RMSE_array.append(sum(RMSE_array_temp)/5)

        std_scores.append(np.std(scores_temp))
        std_times.append(np.std(times_temp))
        std_RMSE.append(np.std(RMSE_array_temp))

    csv_input = pd.read_csv('reg.csv')
    csv_input['Uniform_RMSE'] = RMSE_all
    csv_input['Uniform_time'] = times_all
    csv_input.to_csv('reg.csv', index=False)

    print("data size:",coreset_size)
    print("fitting times for uniform random data: ", times)
    print("std times for uniform random data: ", std_times)
    print("RMSE for uniform random data:", RMSE_array)
    print("std RMSE for uniform random data:", std_RMSE)
    print("scores for uniform reandom data:" , scores)
    print("std scores for uniform reandom data:" , std_scores)



def print_results (df, origin_data_time, origin_data_score , coreset_time, std_coreset_time,  building_coreset_time, std_building_time, coreset_size, std_coreset_size, RMSE_array=[], std_RMSE = []):
    cols = df.columns
    print("#####################################")
    print("shape of coreset:",coreset_size)
    print("std coreset size: ",  std_coreset_size)
    print("size of origin data:", df.shape)
    #print(cols)
    print("Lable to check: ", cols[D - 1])

    print("Total time on the real data: ", origin_data_time)
    print("Total times on the coreset without the construction : ", coreset_time)
    print("std coreset time: ",  std_coreset_time)
    print("building_corset_time: ", building_coreset_time)
    print("std building coreset time:" , std_building_time)
    print("score for original data is: ", origin_data_score, "score after coreset data: ", scores)
    print("RMSE on coreset data:", RMSE_array)
    print( "std RMSE on coreset data: ", std_RMSE)
    print("#####################################")


if __name__== "__main__":
    """
        df,D  =  readData(path = 'D:/datasets/BF.csv')
    vec = create_vec_from_dataset(df)
    n_points, n_weights, n_divs = to_coreset(vec,0.8,2)

    sum_of_weights = np.sum(n_weights)
    weighed_points = []
    weighted_sum = 0
    sum_p=0
    for i in range(len(n_points)):
        weighed_points.append(tuple(j * n_weights[i] for j in n_points[i][1:] )) #u(c)*c for every c in C

    weighted_sum = np.array(weighed_points[0])
    for i in range(1,len(weighed_points)):
        weighted_sum += weighed_points[i]

    origin_points = []
    for i in range (len(vec)):
        origin_points.append(tuple(j for j in vec[i][1:]))

    sum_p= origin_points[0]
    for i in range(1,len(vec)):
        sum_p +=np.array(origin_points[i])

    x = 17
    o_s = 0
    o_c = 0
    for i in range(len(origin_points)):
        o_s += pow(origin_points[i][0]-x,2)

    for i in range(len(n_weights)):
        o_c += n_weights[i]*pow(n_points[i][1]-x,2)

    """
    import csv

    with open('classi_res.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["coreset_size","building_coreset_time" ,"acc_coreset", "acc_origin", "time_coreset", "time_origin","bicreteria_time", "eps"])

        flag = True
        df, D = readData(path = 'D:/datasets/adult.csv')

        # convert nominal data to ordinal data: label encoder
        df = preprocesingAdultData(df)

        times =[]
        scores = []
        times_without_coreset =[]
        times_of_coreset = []
        building_corset_time =[]
        time_of_coreset_without_building = []

        x_train, y_train, x_test, y_test, origin_data_time, origin_data_score = classification_origin_data(df)
        coreset_time, std_coreset_time, building_coreset_time, std_building_time, scores, std_score, coreset_size, std_coreset_size = classification_coreset(df, D, x_train, y_train, x_test, y_test, origin_data_time, origin_data_score, writer)
        print("$$$$$$$$$classification problem")
        print_results(df, origin_data_time, origin_data_score, coreset_time, std_coreset_time, building_coreset_time, std_building_time, coreset_size, std_coreset_size )
        print("$$$$$$$$$classification problem unifrom")
    classification_uniform(df, D,x_train, y_train, x_test, y_test, coreset_size)

    with open('reg.csv', 'w', newline='') as file:
        scores = []
        times = []
        coreset_scores= []
        times_without_coreset =[]
        times_of_coreset = []
        building_corset_time =[]
        time_of_coreset_without_building = []
        RMSE_array= []
        df, D = readData(path = 'D:/datasets/BlackFriday.csv')
        df, D= preprocesingBlackFridayData(df)

        writer = csv.writer(file)
        writer.writerow(["coreset_size","building_coreset_time", "RMSE_coreset", "RMSE_origin", "time_coreset", "time_origin", "bicreteria_time", "eps"])

        x_train, y_train, x_test, y_test, origin_data_time, origin_data_score, RMSE_origin  = regration_origin_data(df)
        coreset_time, std_coreset_time, building_coreset_time, std_building_time, scores, std_score, coreset_size, std_coreset_size, RMSE_array, std_RMSE = regration_coreset(df,D,x_train, y_train, x_test, y_test, RMSE_origin, origin_data_time, writer)
        print_results(df, origin_data_time, origin_data_score, coreset_time, std_coreset_time, building_coreset_time, std_building_time, coreset_size, std_coreset_size, RMSE_array, std_RMSE )

        RMSE_array = []
        scores = []
        times = []
    regration_for_uniform(df, D, x_train, y_train, x_test, y_test, coreset_size)








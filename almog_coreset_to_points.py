import almog_bicriteria as bcr
import numpy as np


# input: node - the coreset tree's root, d - the dimension of the points, eps
# output: normalized weighted points from the original points, with the coreset's points' labels
def coreset_to_points(node, d, eps):
    p_list, w_list = coreset_to_points_rec(node, d, eps)

    print("NUMBER OF OUTPUT POINTS: ", len(p_list))
    # normalize the points
    p_amount = calc_p_amount(node)
    p_sum = sum(w_list)
    for i in range(len(w_list)):
        w_list[i] = (w_list[i] / p_sum) * p_amount
    return p_list, w_list


# the recursive part of coreset_to_points
def coreset_to_points_rec(node, d, eps):
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
        p_idx = 0
        # the extra weight (between 0 and 1) being "carried" between one sequence and the next in case not all of the
        # weights are natural numbers
        i = 0
        while i < cp_amount:
            if i == cp_amount - 1:
                # the number in p_amount[i][0] should be a natural with a small positive/negative noise.
                # get rid of the noise with round
                p_amount[i][0] = round(p_amount[i][0])
            cara_p_list = []
            # add new points with the label of the current coreset point
            while p_amount[i][0] >= 1:
                # copy one of the original points and change it's label
                new_point = list(node.points[p_idx]).copy()
                new_point[d] = node.points[node.coreset.cara_idx[i]][d]
                # append the point into the list and give it a current weight of 1
                if(p_idx in node.coreset.cara_idx):
                    cara_p_list.append(new_point)
                # next point
                p_idx += 1
                p_amount[i][0] -= 1
            # add a "subpoint" at the end, with the label being the weighted average of the current label and the next
            if i < cp_amount - 1 and p_amount[i][0] != 0 and p_idx in node.coreset.cara_idx:
                new_point = list(node.points[p_idx]).copy()
                # the extra weight computation
                extra_weight = [p_amount[i][0]]
                e_p_labels = [node.points[node.coreset.cara_idx[i]][d]]
                while sum(extra_weight) < 1 and i < cp_amount - 1:
                    if sum(extra_weight) + p_amount[i + 1][0] <= 1:
                        # the next cara point weight is small, we will insert it to this point
                        extra_weight += [p_amount[i + 1][0]]
                        p_amount[i + 1][0] = 0
                        e_p_labels += [node.points[node.coreset.cara_idx[i + 1]][d]]
                        i += 1
                    else:
                        # the next cara point weight is big, just add a part of it to this point
                        p_amount[i + 1][0] -= 1 - sum(extra_weight)
                        extra_weight += [1 - sum(extra_weight)]
                        e_p_labels += [node.points[node.coreset.cara_idx[i + 1]][d]]
                # change the label to the weighted average
                new_point[d] = 0
                for j, e_w in enumerate(extra_weight):
                    new_point[d] += e_w * e_p_labels[j]
                # append the point into the list and give it a current weight of 1
                new_point = tuple(new_point)
                cara_p_list.append(new_point)
                p_idx += 1
            # weights calculation
            cara_p_list, cara_w_list = get_p_weights(cara_p_list, d)
            # using the bicriteria to choose representatives
            if len(cara_p_list) > 0:
                cara_p_list, cara_w_list = bcr.get_bicriteria_representatives(cara_p_list, cara_w_list, eps)
                p_list = p_list + cara_p_list
                w_list = w_list + cara_w_list
            i += 1
    # recursion- calling recursive function until we reach the leafs of the tree
    if len(node.subparts) > 0:
        for n in node.subparts:
            new_p_list, new_w_list = coreset_to_points_rec(n, d, eps)
            p_list += new_p_list
            w_list += new_w_list

    return p_list, w_list


# input: orig_points - the unweighted points, d - the dimension of the points
# output: the points and their weights
def get_p_weights(orig_points, d):
    sorted_p = [list(p) for p in orig_points]
    # add a value in the list for a new weight for each point
    for j in range(len(sorted_p)):
        # new weights at cell d+1
        sorted_p[j] += [1]
    for dim in range(d):
        # sort by current dimension
        sorted_p = sorted(sorted_p, key=lambda k: [k[dim]])
        # computing the new weights
        for i in range(len(sorted_p)):
            sorted_p[i][d+1] = sorted_p[i][d+1] * max(i, len(sorted_p)-i)
    sorted_p.sort()
    new_weights = []
    # splitting the points into separate points and weights
    for i in range(len(sorted_p)):
        new_weights.append(sorted_p[i][d+1])
        del sorted_p[i][-1]  # delete cell d+1 of new weights
    # make the list of points back into a tuple
    sorted_p = [tuple(p) for p in orig_points]
    return sorted_p, new_weights


# input: a root of a coreset division tree
# output: the amount of points in the tree
def calc_p_amount(root):
    if len(root.points) > 0:
        return len(root.points)
    if len(root.subparts) > 0:
        p_amnt = 0
        for n in root.subparts:
            p_amnt += calc_p_amount(n)
        return p_amnt

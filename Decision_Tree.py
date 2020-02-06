import csv
import sys
import os

def str_column_to_int(Input_list, i):
    for row in Input_list:
        row[i] = int(row[i].strip())


def make_prediction(node, row):

    if row[node['field']] < node['value']:
        # predicting which class the data will belong to
        if type(node['left_branch']) is dict:
            return make_prediction(node['left_branch'], row)
        else:
            return node['left_branch']
    else:
        if type(node['right_branch']) is dict:
            return make_prediction(node['right_branch'], row)
        else:
            return node['right_branch']


def group_at_split(Input_list, field_num, value):
    # creating group at the plit node
    left_branch, right_branch = list(), list()
    for row in Input_list:
        if row[field_num] < value:
            left_branch.append(row)
        else:
            right_branch.append(row)
    return left_branch, right_branch


def terminal_group_class(group_of_items, fields):
    # deterimining what class a terminal group of data items will belong to
    item_classes = list()
    max = 0
    group_class = None
    for row in group_of_items:
        item_classes.append(row[fields - 1])
    for iterator in item_classes:
        freq = item_classes.count(iterator)
        if max < freq:
            max = freq
            group_class = iterator
    return group_class


def calculate_gini_index(split_group, classes, fields):
    # calculating gini inex
    total_frequency = 0;
    for group in split_group:
        total_frequency += len(group)
    total_frequency = float(total_frequency)
    gini_index = 0.0
    for group in split_group:
        group_size = len(group)
        if group_size == 0:
            continue
        group_gini = 0.0
        for class_id in classes:
            occurences = [row[fields - 1] for row in group].count(class_id)
            occurences = occurences / group_size
            group_gini = occurences ** 2
        gini_index += (1.0 - group_gini) * (group_size / total_frequency)
    return gini_index


def find_split_point(Input_list, fields):
    # determining at what value the group of nodes will be split
    classes = []
    for row in Input_list:
        classes.append(row[fields - 1])
    classes = set(classes)
    classes = list(classes)
    temp_score, temp_value, temp_index, temp_grouping = float('inf'), float('inf'), float('inf'), None
    for field_num in range(fields - 1):
        for row in Input_list:
            split_group = group_at_split(Input_list, field_num, row[field_num])
            current_gini_index = calculate_gini_index(split_group, classes, fields)
            if current_gini_index < temp_score:
                temp_index, temp_value, temp_score, temp_grouping = field_num, row[
                    field_num], current_gini_index, split_group
    split_node = {}
    split_node['field'] = temp_index
    split_node['value'] = temp_value
    split_node['branch_grouping'] = temp_grouping
    return split_node


def split_at_current_node(splitting_node, max_depth, min_reecords, tree_depth, fields):
    # splitting the value at the current decision node
    left_branch = splitting_node['branch_grouping'][0]
    right_branch = splitting_node['branch_grouping'][1]
    del (splitting_node['branch_grouping'])
    if len(left_branch) == 0 or len(right_branch) == 0:
        res = terminal_group_class(left_branch + right_branch, fields)
        splitting_node['left_branch'] = splitting_node['right_branch'] = res
        return
    if tree_depth >= max_depth:
        splitting_node['left_branch'] = terminal_group_class(left_branch, fields)
        splitting_node['right_branch'] = terminal_group_class(right_branch, fields)
        return
    if len(left_branch) <= min_reecords:
        splitting_node['left_branch'] = terminal_group_class(left_branch, fields)
    else:
        splitting_node['left_branch'] = find_split_point(left_branch, fields)
        split_at_current_node(splitting_node['left_branch'], max_depth, min_reecords, tree_depth + 1, fields)
    if len(right_branch) <= min_reecords:
        splitting_node['right_branch'] = terminal_group_class(left_branch, fields)
    else:
        splitting_node['right_branch'] = find_split_point(right_branch, fields)
        split_at_current_node(splitting_node['right_branch'], max_depth, min_reecords, tree_depth + 1, fields)


def build_decision_tree(Input_list, max_depth, min_records, fields):
    # building the decision tree
    root = find_split_point(Input_list, fields)
    split_at_current_node(root, max_depth, min_records, 1, fields)
    return root


def calculate_accuracy(test_results, accurate_results):
    correct_predictions = 0
    for i in range(len(accurate_results)):
        if accurate_results[i] == test_results[i]:
            correct_predictions += 1
    return (correct_predictions/len(accurate_results)) *100


if __name__ == '__main__':

    Input_file = os.path.basename(sys.argv[1])
    Test_data_file = os.path.basename(sys.argv[2])

    Input_list = []
    Test_data = []
    Accurate_results = []
    # taking input from train csv file
    with open(Input_file) as csvfile:

        readCSV = csv.reader(csvfile, delimiter=',') #os.path.basename(path)
        for row in readCSV:
            Input_list.append(row)
    # formatting input for processing
    fields = len(Input_list[0])
    for i in range(fields):
        str_column_to_int(Input_list, i)
    # taking input from test csv file
    with open(Test_data_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            Test_data.append(row)
    test_fields = len(Test_data[0])
    # formatting the test data
    for i in range(test_fields):
        str_column_to_int(Test_data, i)
    pseudo_accurate_results = []
    # accuracy_test_file = Input_file[:11] + 'example_predictions.csv'
    # with open(accuracy_test_file) as csvfile:
    #     readCSV = csv.reader(csvfile, delimiter=',')
    #     for row in readCSV:
    #         pseudo_accurate_results.append(row)
    #
    # accuracy_fields = len(pseudo_accurate_results[0])
    # for i in range(accuracy_fields):
    #     str_column_to_int(pseudo_accurate_results, i)
    #
    # for i in range(len(pseudo_accurate_results)):
    #     Accurate_results.append(pseudo_accurate_results[i][0])

    test_results = []

    root = build_decision_tree(Input_list, 100, 3, fields)
    for row in Test_data:
        test_results.append(make_prediction(root, row))


    # accuracy = calculate_accuracy(test_results, Accurate_results)
    # print(accuracy)
    # creating the output csv file

    i = Input_file.find('_')

    output_file = 'blackbox'+Input_file[8:10] + '_predictions.csv'
    output_list = []
    for i in range(len(test_results)):
        l = list()
        l.append(test_results[i])
        output_list.append(l)
        # writeFile = open(output_file, 'w')
        # writer= csv.writer(writeFile)
    #     writing to the output csv file
    with open(output_file, 'w') as writeFile:
        writer = csv.writer(writeFile)
        for i in range(len(test_results)):
            l = list()
            l.append(test_results[i])
            writer.writerow(l)
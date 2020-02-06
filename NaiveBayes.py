import math
import os
import sys
from Blackbox31 import blackbox31
# from Blackbox32 import blackbox32

# Maintaining the prior probabilities, means and variances of different variables
prior_prob_0 = 0.0
prior_prob_1 = 0.0
prior_prob_2 = 0.0
seen_class_0 = 0
seen_class_1 = 0
seen_class_2 = 0
total_input = 0
mean_X_for_0 = 0
mean_X_for_1 = 0
mean_X_for_2 = 0
mean_Y_for_0 = 0
mean_Y_for_1 = 0
mean_Y_for_2 = 0
mean_Z_for_0 = 0
mean_Z_for_1 = 0
mean_Z_for_2 = 0
var_X_for_0 = 0
var_X_for_1 = 0
var_X_for_2 = 0
var_Y_for_0 = 0
var_Y_for_1 = 0
var_Y_for_2 = 0
var_Z_for_0 = 0
var_Z_for_1 = 0
var_Z_for_2 = 0


def Calculate_0_atributes(parameters):

    global seen_class_0, mean_X_for_0, mean_Y_for_0, mean_Z_for_0, var_X_for_0, var_Y_for_0, var_Z_for_0, X_list_0, Y_list_0, Z_list_0, special_variance
    # Calculating the means of X, Y and Z for 0
    previous_mean_X = mean_X_for_0
    previous_mean_Y = mean_Y_for_0
    previous_mean_Z = mean_Z_for_0
    mean_X_for_0 = (mean_X_for_0 * (seen_class_0 - 1) + parameters[0]) / seen_class_0
    mean_Y_for_0 = (mean_Y_for_0 * (seen_class_0 - 1) + parameters[1]) / seen_class_0
    mean_Z_for_0 = (mean_Z_for_0 * (seen_class_0 - 1) + parameters[2]) / seen_class_0

    # Calculating the variances of X, Y and Z for 0
    if seen_class_0 < 2:
        var_X_for_0 = 0
        var_Y_for_0 = 0
        var_Z_for_0 = 0
    else:
        var_X_for_0 = (((seen_class_0 - 2) / (seen_class_0 - 1)) * var_X_for_0) + (
                    ((parameters[0] - previous_mean_X) ** 2) / seen_class_0)
        var_Y_for_0 = (((seen_class_0 - 2) / (seen_class_0 - 1)) * var_Y_for_0) + (
                    ((parameters[1] - previous_mean_Y) ** 2) / seen_class_0)
        var_Z_for_0 = (((seen_class_0 - 2) / (seen_class_0 - 1)) * var_Z_for_0) + (
                    ((parameters[2] - previous_mean_Z) ** 2) / seen_class_0)


def Calculate_1_atributes(parameters):
    global seen_class_1, mean_X_for_1, mean_Y_for_1, mean_Z_for_1, var_X_for_1, var_Y_for_1, var_Z_for_1, X_list_1, Y_list_1, Z_list_1
    # Calculating the means of X, Y and Z for 1
    previous_mean_X = mean_X_for_1
    previous_mean_Y = mean_Y_for_1
    previous_mean_Z = mean_Z_for_1
    mean_X_for_1 = (mean_X_for_1 * (seen_class_1 - 1) + parameters[0]) / seen_class_1
    mean_Y_for_1 = (mean_Y_for_1 * (seen_class_1 - 1) + parameters[1]) / seen_class_1
    mean_Z_for_1 = (mean_Z_for_1 * (seen_class_1 - 1) + parameters[2]) / seen_class_1

    # Calculating the variances of X, Y and Z for 1
    if seen_class_1 < 2:
        var_X_for_1 = 0
        var_Y_for_1 = 0
        var_Z_for_1 = 0
    else:
        var_X_for_1 = (((seen_class_1 - 2) / (seen_class_1 - 1)) * var_X_for_1) + (
                    ((parameters[0] - previous_mean_X) ** 2) / seen_class_1)
        var_Y_for_1 = (((seen_class_1 - 2) / (seen_class_1 - 1)) * var_Y_for_1) + (
                    ((parameters[1] - previous_mean_Y) ** 2) / seen_class_1)
        var_Z_for_1 = (((seen_class_1 - 2) / (seen_class_1 - 1)) * var_Y_for_2) + (
                    ((parameters[2] - previous_mean_Z) ** 2) / seen_class_1)


def Calculate_2_atributes(parameters):
    global seen_class_2, mean_X_for_2, mean_Y_for_2, mean_Z_for_2, var_X_for_2, var_Y_for_2, var_Z_for_2, X_list_2, Y_list_2, Z_list_2
    # Calculating the means of X, Y and Z for 2
    previous_mean_X = mean_X_for_2
    previous_mean_Y = mean_Y_for_2
    previous_mean_Z = mean_Z_for_2
    mean_X_for_2 = (mean_X_for_2 * (seen_class_2 - 1) + parameters[0]) / seen_class_2
    mean_Y_for_2 = (mean_Y_for_2 * (seen_class_2 - 1) + parameters[1]) / seen_class_2
    mean_Z_for_2 = (mean_Z_for_2 * (seen_class_2 - 1) + parameters[2]) / seen_class_2

    # Calculating the variances of X, Y and Z for 2
    if seen_class_2 < 2:
        var_X_for_2 = 0
        var_Y_for_2 = 0
        var_Z_for_2 = 0
    else:
        var_X_for_2 = (((seen_class_2 - 2) / (seen_class_2 - 1)) * var_X_for_2) + (
                    ((parameters[0] - previous_mean_X) ** 2) / seen_class_2)
        var_Y_for_2 = (((seen_class_2 - 2) / (seen_class_2 - 1)) * var_Y_for_2) + (
                    ((parameters[1] - previous_mean_Y) ** 2) / seen_class_2)
        var_Z_for_2 = (((seen_class_2 - 2) / (seen_class_2 - 1)) * var_Z_for_2) + (
                    ((parameters[2] - previous_mean_Z) ** 2) / seen_class_2)


def Gausssian_distribution_0(parameter):
    global prior_prob_0, seen_class_0, mean_X_for_0, mean_Y_for_0, mean_Z_for_0, var_X_for_0, var_Y_for_0, var_Z_for_0, X_list_0, Y_list_0, Z_list_0
    # Calculating the probabilities of X, Y and Z given the class value is 0
    if var_X_for_0 == 0:
        prob_X_0 = (math.exp(-1 * math.pow(parameter[0] - mean_X_for_0, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_X_0 = (math.exp(-1 * math.pow(parameter[0] - mean_X_for_0, 2) / (2 * var_X_for_0))) / (
            math.sqrt(2 * math.pi * var_X_for_0))

    if var_Y_for_0 == 0:
        prob_Y_0 = (math.exp(-1 * math.pow(parameter[1] - mean_Y_for_0, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_Y_0 = (math.exp(-1 * math.pow(parameter[1] - mean_Y_for_0, 2) / (2 * var_Y_for_0))) / (
            math.sqrt(2 * math.pi * var_Y_for_0))

    if var_Z_for_0 == 0:
        prob_Z_0 = (math.exp(-1 * math.pow(parameter[0] - mean_Z_for_0, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_Z_0 = (math.exp(-1 * math.pow(parameter[2] - mean_Z_for_0, 2) / (2 * var_Z_for_0))) / (
            math.sqrt(2 * math.pi * var_Z_for_0))
    # Calculating the probability of the new data instance belonging to class 0
    prob_0 = prior_prob_0 * prob_X_0 * prob_Y_0 * prob_Z_0
    return prob_0


def Gausssian_distribution_1(parameter):
    global prior_prob_1, seen_class_1, mean_X_for_1, mean_Y_for_1, mean_Z_for_1, var_X_for_1, var_Y_for_1, var_Z_for_1, X_list_1, Y_list_1, Z_list_1
    # Calculating the probabilities of X, Y and Z given the class value is 1
    if var_X_for_1 == 0:
        prob_X_1 = (math.exp(-1 * math.pow(parameter[0] - mean_X_for_1, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_X_1 = (math.exp(-1 * math.pow(parameter[0] - mean_X_for_1, 2) / (2 * var_X_for_1))) / (
            math.sqrt(2 * math.pi * var_X_for_1))

    if var_Y_for_1 == 0:
        prob_Y_1 = (math.exp(-1 * math.pow(parameter[1] - mean_Y_for_1, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_Y_1 = (math.exp(-1 * math.pow(parameter[1] - mean_Y_for_1, 2) / (2 * var_Y_for_1))) / (
            math.sqrt(2 * math.pi * var_Y_for_1))

    if var_Z_for_1 == 0:
        prob_Z_1 = (math.exp(-1 * math.pow(parameter[0] - mean_Z_for_1, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_Z_1 = (math.exp(-1 * math.pow(parameter[2] - mean_Z_for_1, 2) / (2 * var_Z_for_1))) / (
            math.sqrt(2 * math.pi * var_Z_for_1))
    # Calculating the probability of the new data instance belonging to class 1
    prob_1 = prior_prob_1 * prob_X_1 * prob_Y_1 * prob_Z_1
    return prob_1


def Gausssian_distribution_2(parameter):
    global prior_prob_2, seen_class_2, mean_X_for_2, mean_Y_for_2, mean_Z_for_2, var_X_for_2, var_Y_for_2, var_Z_for_2, X_list_2, Y_list_2, Z_list_2
    # Calculating the probabilities of X, Y and Z given the class value is 2
    if var_X_for_2 == 0:
        prob_X_2 = (math.exp(-1 * math.pow(parameter[0] - mean_X_for_2, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_X_2 = (math.exp(-1 * math.pow(parameter[0] - mean_X_for_2, 2) / (2 * var_X_for_2))) / (
            math.sqrt(2 * math.pi * var_X_for_2))

    if var_Y_for_2 == 0:
        prob_Y_2 = (math.exp(-1 * math.pow(parameter[1] - mean_Y_for_2, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_Y_2 = (math.exp(-1 * math.pow(parameter[1] - mean_Y_for_2, 2) / (2 * var_Y_for_2))) / (
            math.sqrt(2 * math.pi * var_Y_for_2))

    if var_Z_for_2 == 0:
        prob_Z_2 = (math.exp(-1 * math.pow(parameter[0] - mean_Z_for_2, 2) / (2 * 1e-9))) / (
            math.sqrt(2 * math.pi * 1e-9))
    else:
        prob_Z_2 = (math.exp(-1 * math.pow(parameter[2] - mean_Z_for_2, 2) / (2 * var_Z_for_2))) / (
            math.sqrt(2 * math.pi * var_Z_for_2))
    # Calculating the probability of the new data instance belonging to class 2
    prob_2 = prior_prob_2 * prob_X_2 * prob_Y_2 * prob_Z_2
    return prob_2


if __name__ == '__main__':

    Input_file = os.path.basename(sys.argv[-1])

    Input_list = list()
    test_list = list()
    result_list = list()
    accuracy_check = list()
    X_points = list()
    Y_points = list()
    Output_list = list()
    if Input_file == 'blackbox31':
        bb = blackbox31
    elif Input_file == 'blackbox32':
        bb = blackbox32

    for i in range(0, 200):
        test_list.append(bb.ask())

    for i in range(0, 100):
        for j in range(0, 10):
            Input = bb.ask()
            Input_list.append(Input)
            total_input += 1
            if Input[1] == 0:
                seen_class_0 += 1
                Calculate_0_atributes(Input[0])
            elif Input[1] == 1:
                seen_class_1 += 1
                Calculate_1_atributes(Input[0])
            elif Input[1] == 2:
                seen_class_2 += 1
                Calculate_2_atributes(Input[0])
            # Calculating the Prior Probabilities of classes 0, 1 and 2
            prior_prob_0 = (seen_class_0) / (total_input)
            prior_prob_1 = (seen_class_1) / (total_input)
            prior_prob_2 = (seen_class_2) / (total_input)
        for k in test_list:
            result_list.clear()
            result_list.append(Gausssian_distribution_0(k[0]))
            result_list.append(Gausssian_distribution_1(k[0]))
            result_list.append(Gausssian_distribution_2(k[0]))

            l = result_list.index(max(result_list))
            tup = k[1], l
            accuracy_check.append(tup)

        total_cases = 0
        correct_cases = 0
        for i in accuracy_check:
            total_cases += 1
            if i[0] == i[1]:
                correct_cases += 1
        accuracy = correct_cases / total_cases
        accuracy = float("{0:.3f}".format(accuracy))
        str1 = str(total_input) + ', ' + str(accuracy)
        Output_list.append(str1)

    Output_file = 'results_' + Input_file + '.txt'
    file = open(Output_file, "w+")
    for str in Output_list:
        file.write(str + '\n')



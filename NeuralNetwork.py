import csv
import numpy as np
import sys
import os

from utils import gen_batches
from utils import relu
from utils import softmax
from utils import log_loss
from utils import label_binarize
from utils import AdamOptimizer
import copy
from utils import accuracy_score
import matplotlib.pyplot as mp

def str_column_to_int(Input_list, i):
    for row in Input_list:
        row[i] = int(row[i].strip())
EPOCHS = 10
LEARNING_RATE = .001
BATCH_SIZE = 100
LAYER_CONF = 0

class NeuralNetwork:
    def __init__(self,layers):

        self.num_layers = len(layers);
        self.biases_between_input_and_hidden1 = np.random.randn(layers[1],1)
        self.biases_between_hidden1_and_hidden2 = np.random.randn(layers[2],1)
        self.biases_between_hidden2_and_output = np.random.randn(layers[3],1)
        self.weights_between_input_and_hidden1 = np.random.randn(layers[1],layers[0]) * np.sqrt(2/layers[0])
        self.weights_between_hidden1_and_hidden2 =  np.random.randn(layers[2],layers[1])* np.sqrt(2/layers[1])
        self.weights_between_hidden2_and_output = np.random.randn(layers[3],layers[2])* np.sqrt(2/layers[2])
        self.loss = 0




        self.layer1_output = []
        self.layer2_output = []
        self.final_output = []
        self.weights1_gradient = []
        self.weights2_gradient = []
        self.weights3_gradient = []
        self.bias1_gradient = []
        self.bias2_gradient = []
        self.bias3_gradient = []
        self.input_hidden2_derivative = np.array(self.weights_between_hidden2_and_output).T
        self.input_hidden1_dervative = np.array(self.weights_between_hidden1_and_hidden2).T
        self.input_layer_deravtive = np.array(self.weights_between_input_and_hidden1).T
        self.params = []
        self.gradients = []
        self.predicted_output = []
        self.accurate_classes = []
        self.training_accuracies =[]
        self.testing_accuracies = []
        self.y_axis = []
        self.x_axis = []

    def clear_lists(self):
        self.layer1_output = list()
        self.layer2_output = list()
        self.final_output = list()
        self.weights1_gradient = list()
        self.weights2_gradient = list()
        self.weights3_gradient = list()
        self.bias1_gradient = list()
        self.bias2_gradient = list()
        self.bias3_gradient = list()
        self.input_hidden2_derivative = np.array(self.weights_between_hidden2_and_output).T
        self.input_hidden1_dervative = np.array(self.weights_between_hidden1_and_hidden2).T
        self.input_layer_deravtive = np.array(self.weights_between_input_and_hidden1).T
        self.params = list()
        self.gradients = list()

        self.layer1_output.clear()
        self.layer2_output.clear()
        self.final_output.clear()
        self.weights1_gradient.clear()
        self.weights2_gradient.clear()
        self.weights3_gradient.clear()
        self.bias1_gradient.clear()
        self.bias2_gradient.clear()
        self.bias3_gradient.clear()
        self.params.clear()
        self.gradients.clear()

    def str_column_to_int(self,Input_list, i):
        for row in Input_list:
            row[i] = int(row[i].strip())

    def feed_forward(self,weights,inputs,biases):

        inputs = np.array(inputs,ndmin=2).T

        dot_in = np.dot(weights, inputs) + biases

        soft_in = list()
        for i in dot_in:
            soft_in.append(i.item(0))

        result = relu(np.array([soft_in]))

        return result[0]


    def get_softmax_output(self,weights,inputs,biases):

        inputs = np.array(inputs, ndmin=2).T

        dot_in = np.dot(weights,inputs) + biases

        soft_in = list()
        for i in dot_in:
            soft_in.append(i.item(0))

        result = softmax(np.array([soft_in]))

        return result[0]




    def slice_into_batches(self,Input_list):
        produced_batches = [];
        length = len(Input_list)
        for each in gen_batches(length,100):
            produced_batches.append(Input_list[each])
        return produced_batches
    #
    #
    # def get_output(self, input_array):
    #     result =  self.feed_forward(self.weights_between_input_and_hidden1,input_array,self.biases_between_input_and_hidden1)
    #     result = self.feed_forward(self.weights_between_hidden1_and_hidden2,result,self.biases_between_hidden1_and_hidden2)
    #     return self.get_softmax_output(self.weights_between_hidden2_and_output,result,self.biases_between_hidden2_and_output)

    def train(self,Input_list):

        for i in Input_list:
            array = np.asarray(i)

            result = self.feed_forward(self.weights_between_input_and_hidden1,array,self.biases_between_input_and_hidden1)
            self.layer1_output.append(result)

        for i in self.layer1_output:
            result = self.feed_forward(self.weights_between_hidden1_and_hidden2,i,self.biases_between_hidden1_and_hidden2)
            self.layer2_output.append(result)

        for i in self.layer2_output:
            result = self.get_softmax_output(self.weights_between_hidden2_and_output,i,self.biases_between_hidden2_and_output)
            self.final_output.append(result)


    def predict(self,Input_list):
        for i in Input_list:
            array = np.asarray(i)
            result = self.feed_forward(self.weights_between_input_and_hidden1,array,self.biases_between_input_and_hidden1)
            self.layer1_output.append(result)
        for i in self.layer1_output:
            result = self.feed_forward(self.weights_between_hidden1_and_hidden2,i,self.biases_between_hidden1_and_hidden2)
            self.layer2_output.append(result)
        for i in self.layer2_output:
            result = self.get_softmax_output(self.weights_between_hidden2_and_output,i,self.biases_between_hidden2_and_output)
            self.final_output.append(result)

    def back_propagate(self,result,train_data):

        gradient = self.final_output - result
        self.layer2_output = np.asarray(self.layer2_output)
        gradient1 = np.array(gradient, ndmin=2).T
        self.weights3_gradient = np.dot(gradient1, self.layer2_output)

        self.bias3_gradient = (gradient1.T * self.biases_between_hidden2_and_output.T).T
        self.bias3_gradient = np.sum(self.bias3_gradient,axis=1)


        weights3_t = np.array(self.weights_between_hidden2_and_output,ndmin=2).T
        cost_wrt_A2_derivative = np.dot(weights3_t, gradient1)
        output2_t = np.array(self.layer2_output,ndmin=2).T
        hidden2_derivative = output2_t *(1-output2_t)
        loss_derivative_hidden2 = cost_wrt_A2_derivative * hidden2_derivative
        self.layer1_output = np.array(self.layer1_output)
        self.weights2_gradient = np.dot(loss_derivative_hidden2,self.layer1_output)
        self.bias2_gradient = (loss_derivative_hidden2.T * self.biases_between_hidden1_and_hidden2.T).T
        self.bias2_gradient = np.sum(self.bias2_gradient, axis=1)



        weights2_t = np.array(self.weights_between_hidden1_and_hidden2,ndmin=2).T
        cost_wrt_A1_derivative = np.dot(weights2_t, loss_derivative_hidden2)
        output1_t =  np.array(self.layer1_output,ndmin=2).T
        hidden1_derivative = output1_t*(1-output1_t)
        loss_derivative_hidden1 = cost_wrt_A1_derivative *hidden1_derivative
        train_data = np.asarray(train_data)
        self.weights1_gradient = np.dot(loss_derivative_hidden1,train_data)
        self.bias1_gradient = (loss_derivative_hidden1.T*self.biases_between_input_and_hidden1.T).T
        self.bias1_gradient = np.sum(self.bias1_gradient,axis=1)

        self.params.append(self.weights_between_hidden2_and_output)
        self.gradients.append(self.weights3_gradient)
        self.params.append(self.weights_between_hidden1_and_hidden2)
        self.gradients.append(self.weights2_gradient)
        self.params.append(self.weights_between_input_and_hidden1)
        self.gradients.append(self.weights1_gradient)
        self.params.append(self.biases_between_hidden2_and_output.T[0])
        self.gradients.append(self.bias3_gradient)
        self.params.append(self.biases_between_hidden1_and_hidden2.T[0])
        self.gradients.append(self.bias2_gradient)
        self.params.append(self.biases_between_input_and_hidden1.T[0])
        self.gradients.append(self.bias1_gradient)

        optimizer = AdamOptimizer(self.params, learning_rate_init=0.001)
        optimizer.update_params(self.gradients)

    def plot_graph(self,data, x_l='X-axis', y_l='Y-axis', t='Title'):
        mp.plot(data[2],data[0], label = "Testing Accuracies")
        mp.plot(data[2],data[1], label = "Training Accuracies")

        # info = ''
        # for i in range(len(extra_info)):
        #     info += str(extra_info[i]).replace('_', ' - ') + '\n'

        # mp.text(x_pos, y_pos, info, bbox={'alpha': 0.5, 'pad': 10})

        # plot_file_name = 'plots/' + t.replace(' ', '') + '' + test_case_name
        # for i in range(len(extra_name)):
        #     plot_file_name += '_' + str(extra_name[i])
        # plot_file_name += '.
        mp.xlabel('Number of epochs')
        # naming the y axis
        mp.ylabel('Accuracies')
        # giving a title to my graph
        mp.title('Change of accuracies over time')

        # show a legend on the plot
        mp.legend()
        plot_file_name = 'accuracy_vs_number_of_epochs.png'

        mp.savefig(plot_file_name)
        mp.show()





    def emulate_ANN(self,Input_list,test_data,accurate_result,classes):
        train_data = []
        train_data_accurate = []

        for i in Input_list:
            temp = i[:length - 1]
            train_data.append(temp)
            train_data_accurate.append(i[length - 1])


        Training_batches = self.slice_into_batches(train_data)
        Result_batches = self.slice_into_batches(train_data_accurate)


        for i in range(EPOCHS):
            for j in range(len(Training_batches)):
                self.clear_lists()
                self.train(Training_batches[j])
                correct_result_array = np.asarray(Result_batches[j])
                self.final_output = np.array(self.final_output)
                result = label_binarize(correct_result_array,classes)
                # log_loss(result,self.final_output+0.0001)
                self.back_propagate(result, Training_batches[i])
            self.clear_lists()
            self.final_output = list()
            self.final_output.clear()
            self.predicted_output = list()
            self.predicted_output.clear()
            self.accurate_classes = list()
            self.accurate_classes.clear()
            self.predict(train_data)
            self.final_output = np.asarray(self.final_output)
            for i in range(len(self.final_output)):
                self.predicted_output.append(np.argmax(self.final_output[i]))

            self.predicted_output = np.asarray(self.predicted_output)
            self.accurate_classes = np.asarray(train_data_accurate)
            self.accurate_classes = self.accurate_classes.T
            training_accuracy = accuracy_score(self.accurate_classes,self.predicted_output)


            self.clear_lists()
            self.final_output = list()
            self.final_output.clear()
            self.accurate_classes = list()
            self.accurate_classes.clear()
            self.predicted_output = list()
            self.predicted_output.clear()
            self.predict(test_data)
            self.final_output = np.asarray(self.final_output)
            for i in range(len(self.final_output)):
                self.predicted_output.append(np.argmax(self.final_output[i]))
            self.predicted_output = np.asarray(self.predicted_output)
            self.accurate_classes = np.asarray(accurate_result)
            self.accurate_classes = self.accurate_classes.T
            testing_accuracy = accuracy_score(self.accurate_classes[0], self.predicted_output)
            self.training_accuracies.append(training_accuracy)
            self.testing_accuracies.append(testing_accuracy)

        # print(self.training_accuracies)
        # print(self.testing_accuracies)
        # print(type(self.training_accuracies))
        # print(type(self.testing_accuracies))

        for i in range(EPOCHS):
            self.x_axis.append((i+1)*10)
        plot_data = (self.testing_accuracies,self.training_accuracies,self.x_axis)
        #
        #
        self.plot_graph(plot_data,'Epochs', 'Change of accuracy over time', 'Change accuracy per Epoch')
        # print(self.training_accuracies)
        # print(self.testing_accuracies)
        # print(type(self.training_accuracies))
        # print(type(self.testing_accuracies))

        # for i in range(len(self.final_output)):
        #     self.predicted_output.append(np.argmax(self.final_output[i]))
        # self.predicted_output = np.asarray(self.predicted_output)
        # self.accurate_classes = np.asarray(accurate_result)
        # self.accurate_classes = self.accurate_classes.T
        #
        # print(accuracy_score(self.accurate_classes[0],self.predicted_output))

if __name__ == '__main__':

    # Input_file = os.path.basename(sys.argv[1])
    # Test_data_file = os.path.basename(sys.argv[2])

    Input_list = []
    test_data = []
    accurate_result = []
    with open('blackbox23_train.csv') as csvfile:

        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            Input_list.append(row)

    fields = len(Input_list[0])
    for i in range(fields):
        str_column_to_int(Input_list, i)

    with open('blackbox23_test.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            test_data.append(row)

    fields = len(test_data[0])
    for i in range(fields):
        str_column_to_int(test_data, i)
    csvfile.close()

    with open('blackbox23_example_predictions.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            accurate_result.append(row)
    csvfile.close()

    fields = len(accurate_result)
    for i in range(1):
        str_column_to_int(accurate_result, i)
    classes = []
    length = len(Input_list[0])

    for x in Input_list:
        classes.append(x[length-1])
    classes = set(classes)
    total_classes = len(classes)
    classes = list(classes)



    csvfile.close()

    sizes =[];
    sizes.append(length-1)
    sizes.append(10)
    sizes.append(20)
    sizes.append(total_classes)

    ANN = NeuralNetwork(sizes)
    ANN.emulate_ANN(Input_list, test_data,accurate_result,classes)







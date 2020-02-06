import sys
import copy
import os

def Take_Input_And_Initialize(Input_file):
    global Walls, Destination, Walls_list, Rewards_list, Matrix_probabilities, Prob_list, Utility, P, Rp, Df, \
        Non_Reward_Positions, Size, Travel_Policy, Neighbours, Neighbours_list, Non_Terminal_Positions

    Neighbours = {}

    Walls = {}
    Walls_list = []
    Rewards_list = []
    Neighbours = {}
    Matrix_probabilities = {}
    Prob_list = []
    Non_Reward_Positions = []

    f = open(Input_file, "r")

    lines = f.readlines()

    Size = int(lines[0])

    Number_of_walls = int(lines[1])
    for i in range(2, Number_of_walls + 2):
        Wall_position = lines[i].split(",")
        x = int(Wall_position[1])
        y = int(Wall_position[0])
        Walls[(x + 1, y + 1)] = -101
        Walls_list.append((x, y))

    dest_pos = lines[-1].split(",")
    x = int(dest_pos [1] )
    y = int(dest_pos[0] )
    Destination = (x,y)
    Rp = -1
    P = 0.7
    Df = 0.9

    Travel_Policy = [['' for _ in range(Size)] for _ in range(Size)]

    Utility = [[0.0 for _ in range(Size + 2)] for _ in range(Size + 2)]
    for i in range(1,Size + 1):
        for j in range(1, Size +1):
            Utility[i][j] = 0.0

    for i in range(0, len(Walls_list)):
        Travel_Policy[Walls_list[i][0]][Walls_list[i][1]] = "x"

    Utility[Destination[0] + 1][Destination[1] +1] = 99
    Travel_Policy[Destination[0]][Destination[1]] = "."

    for i in range(1, Size + 1):
        for j in range(1, Size + 1):
            if Travel_Policy[i - 1][j - 1] == "" or Travel_Policy[i - 1][j - 1] == "x":
                pos = (i, j)
                Non_Reward_Positions.append(pos)
    Non_Terminal_Positions = set(Non_Reward_Positions)




def Calculate_And_Map_Probabilities():
    for pos in Non_Reward_Positions:
        Probability_list = []
        i = pos[0]
        j = pos[1]

        N = (i - 1, j)
        W = (i, j - 1)
        E = (i, j + 1)
        S = (i + 1, j)

        C_N = 0
        C_S = 0
        C_E = 0
        C_W = 0

        if N[0] == 0:
            N_P = 0
            S_N_P = 0
            E_N_P = 0
            W_N_P = 0
            C_N += 0.7
            C_E += .1
            C_S += .1
            C_W += .1
        else:
            N_P = P
            S_N_P = 0.1
            E_N_P = 0.1
            W_N_P = 0.1

        if S[0] == Size + 1:
            S_P = 0
            N_S_P = 0
            E_S_P = 0
            W_S_P = 0

            C_N += .1
            C_E += .1
            C_S += 0.7
            C_W += .1
        else:
            S_P = P
            N_S_P = 0.1
            E_S_P = 0.1
            W_S_P = 0.1

        if W[1] == 0:
            W_P = 0
            N_W_P = 0
            S_W_P = 0
            E_W_P = 0
            C_N += .1
            C_E += .1
            C_S += 0.1
            C_W += .7
        else:
            W_P = P
            N_W_P = 0.1
            S_W_P = 0.1
            E_W_P = 0.1

        if E[1] == Size + 1:
            E_P = 0
            W_E_P = 0
            N_E_P = 0
            S_E_P = 0
            C_N += .1
            C_E += 0.7
            C_S += .1
            C_W += .1
        else:
            E_P = P
            W_E_P = 0.1
            N_E_P = 0.1
            S_E_P = 0.1

        Probability_list.append(N_P)
        Probability_list.append(N_S_P)
        Probability_list.append(N_E_P)
        Probability_list.append(N_W_P)

        Probability_list.append(S_P)
        Probability_list.append(S_N_P)
        Probability_list.append(S_E_P)
        Probability_list.append(S_W_P)

        Probability_list.append(W_P)
        Probability_list.append(W_E_P)
        Probability_list.append(W_N_P)
        Probability_list.append(W_S_P)

        Probability_list.append(E_P)
        Probability_list.append(E_W_P)
        Probability_list.append(E_N_P)
        Probability_list.append(E_S_P)

        Probability_list.append(C_N)
        Probability_list.append(C_S)
        Probability_list.append(C_W)
        Probability_list.append(C_E)
        for x in range(len(Probability_list)):
            if Probability_list[x] == 0.7999999999999999:
                Probability_list[x] = .8

        Matrix_probabilities[pos] = Probability_list
def Calculate_Utility():
    global loop_count, old_Utility
    Epsilon = ( 0.1 * (1 - Df) )/ Df
    Maximum_Difference = 1.0
    while Maximum_Difference >= Epsilon:
        old_Utility = copy.deepcopy(Utility)
        loop_count += 1
        Maximum_Difference = float("-inf")
        for i in range(1, Size + 1):
            for j in range(1, Size + 1):
                pos = (i,j)
                x = Destination[0]
                y = Destination[1]
                dest = (x + 1, y + 1)

                if not pos == dest:
                    Prob_list = Matrix_probabilities[pos]
                    if pos in Walls:
                        Rp = -101
                    else:
                        Rp = -1

                    N_P = Prob_list[0]
                    N_S_P = Prob_list[1]
                    N_E_P = Prob_list[2]
                    N_W_P = Prob_list[3]

                    S_P = Prob_list[4]
                    S_N_P = Prob_list[5]
                    S_E_P = Prob_list[6]
                    S_W_P = Prob_list[7]

                    W_P = Prob_list[8]
                    W_E_P = Prob_list[9]
                    W_N_P = Prob_list[10]
                    W_S_P = Prob_list[11]

                    E_P = Prob_list[12]
                    E_W_P = Prob_list[13]
                    E_N_P = Prob_list[14]
                    E_S_P = Prob_list[15]


                    C_N = Prob_list[16]
                    C_S = Prob_list[17]
                    C_W = Prob_list[18]
                    C_E = Prob_list[19]

                    N_Utility = N_P * old_Utility[i-1][j] + N_S_P * old_Utility[i+1][j] + N_E_P * old_Utility[i][j + 1] + N_W_P * old_Utility[i][j-1] \
                                + C_N * old_Utility[i][j]

                    S_Utility = S_N_P * old_Utility[i-1][j] + S_P * old_Utility[i + 1][j] + S_E_P * old_Utility[i][j + 1] + S_W_P * old_Utility[i][j - 1] \
                                + C_S * old_Utility[i][j]

                    W_Utility = W_N_P * old_Utility[i - 1][j] + W_S_P * old_Utility[i + 1][j] + W_E_P * old_Utility[i][j + 1] + W_P * old_Utility[i][j - 1]\
                                + C_W * old_Utility[i][j]

                    E_Utility = E_N_P * old_Utility[i - 1][j] + E_S_P * old_Utility[i + 1][j] + E_P * old_Utility[i][j + 1] + E_W_P * old_Utility[i][j - 1]\
                                + C_E * old_Utility[i][j]



                    Max_Utility = N_Utility
                    if S_Utility > Max_Utility:
                        Max_Utility = S_Utility
                    if E_Utility > Max_Utility:
                        Max_Utility = E_Utility
                    if W_Utility > Max_Utility:
                        Max_Utility = W_Utility

                    Old_Utility = old_Utility[i][j]
                    new_Utility = (Max_Utility * Df) \
                                  + Rp



                    Maximum_Difference = max(Maximum_Difference, abs(new_Utility - Old_Utility))
                    Utility[i][j] = new_Utility

def Determine_Travel_Policy():
    for i in range(1, Size + 1):
        for j in range(1, Size + 1):
            pos = (i,j)
            N = (i - 1, j)
            W = (i, j - 1)
            E = (i, j + 1)
            S = (i + 1, j)
            x = Destination[0]
            y = Destination[1]
            dest = (x + 1, y + 1)
            if not pos == dest:
                Prob_list = Matrix_probabilities[pos]

                N_P = Prob_list[0]
                N_S_P = Prob_list[1]
                N_E_P = Prob_list[2]
                N_W_P = Prob_list[3]

                S_P = Prob_list[4]
                S_N_P = Prob_list[5]
                S_E_P = Prob_list[6]
                S_W_P = Prob_list[7]

                W_P = Prob_list[8]
                W_E_P = Prob_list[9]
                W_N_P = Prob_list[10]
                W_S_P = Prob_list[11]

                E_P = Prob_list[12]
                E_W_P = Prob_list[13]
                E_N_P = Prob_list[14]
                E_S_P = Prob_list[15]

                C_N = Prob_list[16]
                C_S = Prob_list[17]
                C_W = Prob_list[18]
                C_E = Prob_list[19]

                N_Utility = N_P * old_Utility[i - 1][j] + N_S_P * old_Utility[i + 1][j] + N_E_P * old_Utility[i][
                    j + 1] + N_W_P * old_Utility[i][j - 1] \
                            + C_N * old_Utility[i][j]

                S_Utility = S_N_P * old_Utility[i - 1][j] + S_P * old_Utility[i + 1][j] + S_E_P * old_Utility[i][
                    j + 1] + S_W_P * old_Utility[i][j - 1] \
                            + C_S * old_Utility[i][j]

                W_Utility = W_N_P * old_Utility[i - 1][j] + W_S_P * old_Utility[i + 1][j] + W_E_P * old_Utility[i][
                    j + 1] + W_P * old_Utility[i][j - 1] \
                            + C_W * old_Utility[i][j]

                E_Utility = E_N_P * old_Utility[i - 1][j] + E_S_P * old_Utility[i + 1][j] + E_P * old_Utility[i][
                    j + 1] + E_W_P * old_Utility[i][j - 1] \
                            + C_E * old_Utility[i][j]
                Max_Utility = N_Utility
                if S_Utility > Max_Utility:
                    Max_Utility = S_Utility
                if E_Utility > Max_Utility:
                    Max_Utility = E_Utility
                if W_Utility > Max_Utility:
                    Max_Utility = W_Utility

                if pos not in Walls:
                    if Max_Utility == N_Utility:
                        Travel_Policy[i - 1][j - 1] = '^'
                    elif Max_Utility == S_Utility:
                        Travel_Policy[i - 1][j - 1] = 'v'
                    elif Max_Utility == E_Utility:
                        Travel_Policy[i - 1][j - 1] = '>'
                    elif Max_Utility == W_Utility:
                        Travel_Policy[i - 1][j - 1] = '<'



if __name__ == '__main__':

    loop_count = 0
    Input_file = sys.argv[1]
    Output_file = sys.argv[2]
    sys.stdout = open(Output_file, 'w')
    Take_Input_And_Initialize(Input_file)
    Calculate_And_Map_Probabilities()
    Calculate_Utility()
    Determine_Travel_Policy()

    for i in range(len(Travel_Policy)):
        str = ""
        for j in range(len(Travel_Policy[i])):
            str += Travel_Policy[i][j]
        print(str, end ='')
        if i != len(Travel_Policy)-1:
            print()
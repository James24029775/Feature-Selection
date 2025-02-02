import argparse
import numpy as np
from numpy import random
import pandas as pd

import matplotlib.pyplot as plt
import os

tasks = ['Individual', 'Set', 'Greedy']
models = ['LR', 'GB', 'DNN']



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--method", default='MD', help="e.g. MD(MaxDelta), MPR(MinPerfReq), MS(MaxScore)", type=str)
    parser.add_argument("-t", "--tolerence", default=0.03, help="Tolerence rate for MPR.", type=float)
    parser.add_argument("-r", "--rho", default=0.003, help="Penalty coefficient for MS.", type=float)
    args = parser.parse_args()

    stop_df_all = []
    index_all = []
    for i in range(len(tasks)):
        for j in range(len(models)):

            filename = tasks[i] + '_F1_' + models[j]
            # read the results, extract their score parts.
            record_df = pd.read_csv('../Results/' + filename + '.csv', index_col=0)
            score_df = pd.DataFrame(index=record_df.index)
            time_df = pd.DataFrame(index=record_df.index)
            size_df = pd.DataFrame(index=record_df.index)
            for column in record_df.columns:
                for index in record_df.index:
                    text_tuple = record_df.loc[index][column]
                    if tasks[i] == 'Set':
                        score, _, size = text2tuple(text_tuple, tasks[i])
                        size_df.at[index, column] = size
                    else:
                        score, _ = text2tuple(text_tuple, tasks[i])
                    score_df.at[index, column] = score
            size_df = size_df.astype(int)

            for index in record_df.index:
                index_score = score_df.loc[index].values
                # Get stopping points by the following methods.
                stop_index1 = MaxDelta(index_score, args, tasks[i], size_df.loc[index])
                stop_index2 = MinPerfReq(index_score, args, tasks[i], size_df.loc[index])
                stop_index3 = MaxScore(index_score, args, tasks[i], size_df.loc[index])
                # stop_index4 = MyScore(index_score, args, tasks[i], size_df.loc[index])

                # save the stopping feature size by different methods
                stop_acc1 = index_score[stop_index1]
                stop_acc2 = index_score[stop_index2]
                stop_acc3 = index_score[stop_index3]
                # stop_acc4 = index_score[stop_index4]
                index_all.append(tasks[i] + '_' + models[j] + '_' + index)
                if tasks[i] == 'Set':
                    info = [size_df.loc[index][str(stop_index1)], stop_acc1, size_df.loc[index][str(stop_index2)], stop_acc2, size_df.loc[index][str(stop_index3)], stop_acc3]
                else:
                    info = [stop_index1+1, stop_acc1, stop_index2+1, stop_acc2, stop_index3+1, stop_acc3]
                stop_df_all.append(info)



    print(len(stop_df_all))
    pd.DataFrame(stop_df_all, index=index_all, columns=['MaxDelta', 'MD_score', 'MinPerfReq', 'MPR_score', 'MaxScore', 'MS_score']).to_csv('../Results/stopping_points.csv')


def text2tuple(text, task):
    info = text[1:-1].split(', ')
    if task == 'Set':
        return float(info[0]), float(info[1]), int(info[2])
    else:
        return float(info[0]), float(info[1])


def MaxDelta(score, args, task, size_df):
    max_delta = 0
    index = len(score) - 1
    for i in range(len(score)-1, 0, -1):
        delta = score[i] - score[i-1]
        if delta > max_delta and score[i-1] != 0:
            max_delta = delta
            index = i

    return index

def MinPerfReq(score, args, task, size_df):
    best_CVscore = score[-1]
    index = len(score) - 1
    for i in range(len(score)-1, 0, -1):
        delta = (best_CVscore - score[i]) / best_CVscore
        if delta > args.tolerence:
            index = i
            break

    return index

def MaxScore(score, args, task, size_df):
    best_performance = 0
    index = len(score) - 1

    for i in range(len(score)-1, 0, -1):
        if task == 'Set':
            adj_score = score[i] - (args.rho * size_df[i])
        else:
            current_size = i + 1
            adj_score = score[i] - (args.rho * current_size)
        if adj_score > best_performance and score[i-1] != 0:
            best_performance = adj_score
            index = i

    return index

def MyScore(score, args, task, size_df):
    smallest_angle = 999
    start_score = score[0]
    end_score = score[len(score)-1]
    index = len(score) - 1
    print(task)
    for i in range(1, len(score)-1):
        start_slope = (score[i] - start_score)/(i - 0)
        end_slope = (end_score - score[i])/(len(score)-1-i)

        x = np.array([1,start_slope])
        y = np.array([1,end_slope])

        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))

        angle = (np.arccos(x.dot(y)/(float(Lx*Ly)))*180/np.pi)
        angle = 180 - angle

        print(i, angle)

        # 
        if smallest_angle > angle and score[i-1] != 0:
            smallest_angle = angle
            index = i
    print(index)
    return index



if __name__ == "__main__":
    main()
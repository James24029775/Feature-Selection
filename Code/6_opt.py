import argparse
import numpy as np
from numpy import random
import pandas as pd

import matplotlib.pyplot as plt
import os

tasks = ['Greedy']
models = ['LR', 'GB', 'DNN']



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--method", default='MD', help="e.g. MD(MaxDelta), MPR(MinPerfReq), MS(MaxScore)", type=str)
    parser.add_argument("-t", "--tolerence", default=0.03, help="Tolerence rate for MPR.", type=float)
    args = parser.parse_args()

    stop_df_all = []
    index_all = []
    rhos = np.arange(0.00000, 0.005, 0.00001)
    rho_df = pd.DataFrame(index=models)
    for rho in rhos:
        for j in range(len(models)):

            filename = tasks[0] + '_F1_' + models[j]
            # read the results, extract their score parts.
            record_df = pd.read_csv('../Results/' + filename + '.csv', index_col=0)
            score_df = pd.DataFrame(index=record_df.index)
            time_df = pd.DataFrame(index=record_df.index)
            size_df = pd.DataFrame(index=record_df.index)
            for column in record_df.columns:
                for index in record_df.index:
                    text_tuple = record_df.loc[index][column]
                    if tasks[0] == 'Set':
                        score, _, size = text2tuple(text_tuple, tasks[0])
                        size_df.at[index, column] = size
                    else:
                        score, _ = text2tuple(text_tuple, tasks[0])
                    score_df.at[index, column] = score
            size_df = size_df.astype(int)

            
            for index in record_df.index:
                index_score = score_df.loc[index].values
                stop_index3 = MaxScore(index_score, rho, tasks[0], size_df.loc[index])

                rho_df.at[models[j], rho] = int(stop_index3)
                
                # if models[j] == 'LR':
                #     #! 0~0.00202
                #     if (8 <= stop_index3+1 and stop_index3+1 <= 10) or (16 <= stop_index3+1 and stop_index3+1 <= 17):
                #         print(tasks[0], models[j], rho, stop_index3+1)
                # elif models[j] == 'GB':
                #     #! 0.00095~0.00371
                #     if stop_index3+1 == 8:
                #         print(tasks[0], models[j], rho, stop_index3+1)
                # elif models[j] == 'DNN':
                #     #! only choose 4 or 22
                #     print(tasks[0], models[j], rho, stop_index3+1)
                #     if 16 <= stop_index3+1 and stop_index3+1 <= 18:
                #         print(tasks[0], models[j], rho, stop_index3+1)
                # #! conclusion: 0.00095~0.00202
                # #! 15, 7, 21

    rho_df.to_csv('../Results/rho_record.csv')

def text2tuple(text, task):
    info = text[1:-1].split(', ')
    if task == 'Set':
        return float(info[0]), float(info[1]), int(info[2])
    else:
        return float(info[0]), float(info[1])



def MaxScore(score, rho, task, size_df):
    best_performance = 0
    index = len(score) - 1

    for i in range(len(score)-1, 0, -1):
        if task == 'Set':
            adj_score = score[i] - (rho * size_df[i])
        else:
            current_size = i + 1
            adj_score = score[i] - (rho * current_size)
        if adj_score > best_performance and score[i-1] != 0:
            best_performance = adj_score
            index = i

    return index




if __name__ == "__main__":
    main()
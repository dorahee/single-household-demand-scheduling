import pandas as pd
import seaborn as sns
import numpy as np

# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later)
date = "0405"
time = "1202"
outType = "out"
file = "results/{0}/{1}/{2}-{0}-{1}.csv".format(date, time, outType)
directory = file[:file.rindex("/") + 1]
data = pd.read_csv(file, header=[0, 1], index_col=[0, 1, 2, 3, 4])
# data = data.xs("time", level="Result").loc[:, (["Gurobi", "Gecode", "OGSA"], slice(None), slice(None))]\
data = data\
           .xs("time", level="Result") \
           .loc[:, (["Model run time ratio"], slice(None), slice(None))] \
           .stack().reset_index()
# data = data.xs("obj", level="Result")["Objective Ratio"].stack()
data.columns = ["", "", "", "", "Solver", "Model run time ratio"]

data2 = np.round(data.groupby("Solver")["Model run time ratio"].describe(), 2)
print(data2)
out_file_name = directory + '/agg-ratio-model-{}-{}.csv'.format(date, time)
data2.to_csv(out_file_name, sep=',', encoding='utf-8')

# data2 = np.round(data.groupby(["Variable choice", "Variable selection", "Model"])["Time (ms)"].describe(), 2)
# print(data2)
# out_file_name = directory + '/time-agg1.csv'
# data2.to_csv(out_file_name, sep=',', encoding='utf-8')
#
# data2 = np.round(data.groupby(["Variable choice", "Model"])["Time (ms)"].describe(), 2)
# print(data2)
# out_file_name = directory + '/time-agg2.csv'
# data2.to_csv(out_file_name, sep=',', encoding='utf-8')
#
# data2 = np.round(data.groupby(["Variable selection", "Model"])["Time (ms)"].describe(), 2)
# print(data2)
# out_file_name = directory + '/time-agg3.csv'
# data2.to_csv(out_file_name, sep=',', encoding='utf-8')

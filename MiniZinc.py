from minizinc import *
import bisect
from numpy import sqrt, pi, random
import numpy as np
import random as r

from bokeh.plotting import figure, show, output_file
from bokeh.models import Legend
from bokeh.layouts import gridplot
from pandas import timedelta_range
# import colorcet as cc
from numpy import genfromtxt
from time import gmtime, strftime, localtime

num_households = 100
num_repeats = 1
colour_step = 1


# create the job
def prices():
    prices2 = genfromtxt('prices.csv', delimiter=',').tolist()
    return prices2


def tasks():
    no_intervals = 144
    no_tasks = 10
    no_intervals_periods = int(no_intervals / 48)

    p_d = genfromtxt('probability.csv', delimiter=',').tolist()

    sum_t = sum(p_d[0])
    p_d_short = [p / sum_t for p in p_d[0]]
    period_options = [i for i in range(48)]

    l_demands = [1.5, 2.3, 3.5, 6, 0.008, 1.1, 2.4, 0.6, 0.5, 0.004, 0.002, 4, 0.6, 0.1, 0.015, 2.4, 0.05, 0.12, 1.2,
                 2.2,
                 0.7, 1.7, 2.1, 0.0015, 0.09, 0.05, 0.01, 0.056, 0.072, 0.65, 2, 1.5, 0.1, 2.4, 1.2, 2.4, 1.2, 1, 0.3,
                 2.4,
                 1.2, 0.075, 0.052, 0.015, 0.045, 0.011, 0.0625, 0.15, 1, 0.005, 1.1, 5, 0.55, 0.1, 0.14, 0.038, 0.035,
                 0.068, 0.072, 0.093, 0.148, 0.7, 0.3, 1, 0.08, 0.12, 0.015, 6, 0.02, 0.075, 0.055, 0.03, 0.13, 0.05,
                 0.21,
                 0.1, 0.005, 1, 3.6, 1.2, 0.9, 1.2, 1.2, 0.05, 0.06, 0.9, 0.4, 2.4, 0.35, 2]

    # I meant mean value is 40 minutes
    mean_value = 40.0 / (24.0 * 60.0 / no_intervals)
    mode_value = sqrt(2 / pi) * mean_value

    # task details
    preferred_starts = []
    earliest_starts = []
    latest_ends = []
    durations = []
    demands = []
    care_factors = []
    predecessors = []
    successors = []
    prec_delays = []
    aggregated_loads = [0] * no_intervals

    for counter_j in range(no_tasks):

        # job consumption per hour
        demand = r.choice(l_demands)
        demands.append(int(demand * 1000))

        # job duration
        duration = int(random.rayleigh(mode_value, 1)[0])
        while duration == 0:
            duration = int(random.rayleigh(mode_value, 1)[0])
        durations.append(duration)

        # job preferred start time
        middle_point = int(np.random.choice(period_options, 1, p=p_d_short)[0] *
                           no_intervals_periods + r.randint(0, 3))
        p_start = (middle_point - int(duration / 2)) % no_intervals
        while p_start + duration - 1 >= no_intervals:
            middle_point = int(
                np.random.choice(period_options, 1, p=p_d_short)[0] * no_intervals_periods + r.randint(0, 3))
            p_start = (middle_point - int(duration / 2)) % no_intervals
        preferred_starts.append(p_start)

        # job earliest starting time
        # e_start = r.choice([i for i in range(-duration + 1, p_start + 1)])
        e_start = 0
        earliest_starts.append(e_start)

        # job latest finish time
        # l_finish = r.choice([i for i in range(p_start - duration + 1, num_intervals - 1 + duration)])
        l_finish = no_intervals - 1
        latest_ends.append(l_finish)

        # job care factor
        care_f = round(r.random(), 1)
        if care_f == 0:
            care_f = 0.01
        care_factors.append(int(care_f * 10))

        if r.choice([True, False]) and counter_j > 0:
            successors.append(counter_j + 1)

            id_predecessor_set = [i for i in range(counter_j)]
            id_predecessor = r.choice(id_predecessor_set)
            predecessors.append(id_predecessor + 1)

            delay = 0 if durations[id_predecessor] + duration >= no_intervals \
                else r.randint(0, no_intervals - durations[id_predecessor] - duration - 1)
            # delay = 144
            prec_delays.append(delay)

        for d in range(duration):
            aggregated_loads[p_start + d] += demand
            # if p_start + k <= num_intervals - 1:
            #     aggregated_loads[p_start + k] += demand
            # else:
            #     aggregated_loads[p_start + k - num_intervals] += demand

    no_precedences = len(predecessors)
    maximum_demand = sum(demands)

    return no_intervals, preferred_starts, no_tasks, earliest_starts, latest_ends, durations, demands, care_factors, \
           no_precedences, predecessors, successors, prec_delays, maximum_demand, aggregated_loads


def solving(model_file, num_intervals, prices, preferred_starts, num_tasks, earliest_starts, latest_ends, durations,
            demands, care_factors, num_precedences, predecessors, successors, prec_delays, max_demand, solve_choice):
    sing_dsp = Instance([model_file])
    sing_dsp["num_intervals"] = num_intervals
    sing_dsp["prices"] = prices
    sing_dsp["preferred_starts"] = preferred_starts
    sing_dsp["num_tasks"] = num_tasks
    sing_dsp["earliest_starts"] = earliest_starts
    sing_dsp["latest_ends"] = latest_ends
    sing_dsp["durations"] = durations
    sing_dsp["demands"] = demands
    sing_dsp["care_factors"] = care_factors
    sing_dsp["num_precedences"] = num_precedences
    sing_dsp["predecessors"] = predecessors
    sing_dsp["successors"] = successors
    sing_dsp["prec_delays"] = prec_delays
    sing_dsp["max_demand"] = max_demand

    solver = load_solver(solve_choice)
    result = solver.solve(sing_dsp)

    return result


# ====== experiments: start ======
# tech = ["mip", "cp"]
tech = ["cp"]

solvers = dict()
solvers["mip"] = ["gurobi"]
# solvers["mip"] = ["gurobi", "cplex"]
solvers["cp"] = ["gecode"]
# solvers["cp"] = ["gecode", "chuffed"]
# solvers["cp"] = ["gecode", "chuffed", "ortools"]

models = dict()
models["mip"] = "./Household-mip.mzn"
models["cp"] = "./Household-cp.mzn"

solveTimes_exp_avg = dict()
household_profiles = []
for e in range(num_households):

    print("exp: {}".format(e))

    num_intervals, preferred_starts, num_tasks, earliest_starts, latest_ends, durations, demands, care_factors, \
    num_precedences, predecessors, successors, prec_delays, max_demand, household_profile = tasks()
    household_profiles.append(household_profile)

    prices_input = prices()

    solveTimes_per_exp = dict()
    for k in range(num_repeats):

        for te in tech:
            model_file = models[te]

            for sol in solvers[te]:

                if e == 0 and k == 0:
                    solveTimes_exp_avg[sol] = []

                if k == 0:
                    solveTimes_per_exp[sol] = []

                res = solving(model_file, num_intervals, prices_input, preferred_starts, num_tasks,
                              earliest_starts, latest_ends, durations, demands, care_factors,
                              num_precedences, predecessors, successors, prec_delays, max_demand, sol)
                obj = res._solutions[-1].objective
                solveTime = res._solutions[-1].statistics['time'].microseconds

                solveTimes_per_exp[sol].append(solveTime)

    for te in tech:
        for sol in solvers[te]:
            solveTimes_exp_avg[sol].append(
                int(sum(solveTimes_per_exp[sol]) / len(solveTimes_per_exp[sol])))
            # print("{:<7}: {}".format(sol, solveTimes_per_exp[sol]))

print("======")
for te in tech:
    for sol in solvers[te]:
        print("{:<7}: {}".format(sol, solveTimes_exp_avg[sol]))

# output run times
output_time = strftime("%m%d_%H%M%S", localtime())
with open('Results/runtimes_{}.csv'.format(output_time), 'w') as f:
    for key in solveTimes_exp_avg.keys():
        f.write("%s,%s\n" % (key, str(solveTimes_exp_avg[key])[1:-1]))

# draw household profile before
plots = []
x_axis = timedelta_range(0, periods=144, freq="10T")
for i, p in enumerate(household_profiles):
    p1 = figure(x_axis_type="datetime", title="Household Demand Profiles")
    p1.grid.grid_line_alpha = 0.3
    p1.yaxis.axis_label = "Demand (KW)"

    # p1.line(x_axis, p, color=palette[i], legend="Household {}".format(i))
    p1.line(x_axis, p, color="blue", legend="Household {}".format(i))
    plots.append(p1)

p2 = figure(x_axis_type="datetime", title="Household Demand Profiles")
p2.grid.grid_line_alpha = 0.3
p2.yaxis.axis_label = "Demand (KW)"
avg_households = np.sum(household_profiles, axis=0)
p2.line(x_axis, avg_households, color="green", legend="Average household")
plots.append(p2)
output_file("Results/runtimes_{}.html".format(output_time))
show(gridplot(plots, ncols=5, plot_width=650, plot_height=450))





# show(p1)

# line = p1.line(x_axis, p, color=palette[i])
# household_lines.append(line)

# legend = Legend(items=[("household {}".format(i), [p]) for i, p in enumerate(household_lines)], location=(0, -30),
#                 orientation="horizontal")
# p1.add_layout(legend, 'below')

# def temp_tasks():
#     num_intervals = 144
#     prices = [141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141,
#               141,
#               141, 141, 141,
#               141, 141, 141, 141, 141, 143, 143, 143, 142, 142, 142, 148, 148, 148, 189, 189, 189, 163,
#               163,
#               163, 145, 145,
#               145, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141,
#               141,
#               141, 141, 141,
#               141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 142, 142, 142, 141, 141, 141,
#               146,
#               146, 146, 158,
#               158, 158, 203, 203, 203, 313, 313, 313, 1155, 1155, 1155, 1155, 1155, 1155, 1446, 1446,
#               1446,
#               616, 616, 616,
#               616, 616, 616, 363, 363, 363, 363, 363, 363, 313, 313, 313, 427, 427, 427, 221, 221, 221,
#               158,
#               158, 158, 148,
#               148, 148, 158, 158, 158, 143, 143, 143, 142, 142, 142, 144, 144, 144]
#     preferred_starts = [41, 104, 89, 26, 24, 67, 99, 49, 16, 42]
#     num_tasks = len(preferred_starts)
#     earliest_starts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     latest_ends = [143, 143, 143, 143, 143, 143, 143, 143, 143, 143]
#     durations = [4, 3, 4, 7, 4, 4, 3, 2, 5, 4]
#     demands = [6000, 15, 5, 2400, 1200, 6000, 80, 210, 210, 11]
#     care_factors = [3, 2, 6, 10, 9, 3, 2, 6, 1, 9]
#     num_precedences = 1
#     predecessors = [8]
#     successors = [9]
#     prec_delays = [129]
#     max_demand = 16131
#
#     return num_intervals, prices, preferred_starts, num_tasks, earliest_starts, latest_ends, durations, \
#            demands, care_factors, num_precedences, predecessors, successors, prec_delays, max_demand
# p_d = [[2722, 2522, 2294, 2086, 1931, 1823, 1768, 1754, 1805, 1931,
#         2229, 2594, 3250, 3699, 4146, 4142, 3829, 3452, 3090, 2791,
#         2548, 2392, 2264, 2268, 2243, 2314, 2389, 2525, 2642, 2857,
#         3127, 3551, 4048, 4446, 4949, 5360, 5396, 5364, 5258, 5238,
#         5206, 4975, 4674, 4376, 4036, 3700, 3466, 3189],
#        [2722, 5244, 7538, 9624, 11555, 13378, 15146, 16900, 18705, 20636,
#         22865, 25459, 28709, 32408, 36554, 40696, 44525, 47977, 51067, 53858,
#         56406, 58798, 61062, 63330, 65573, 67887, 70276, 72801, 75443, 78300,
#         81427, 84978, 89026, 93472, 98421, 103781, 109177, 114541, 119799, 125037,
#         130243, 135218, 139892, 144268, 148304, 152004, 155470, 158659]]

# p_d_short = p_d[1]
#
#
# p_d_long = []
# for i in range(len(p_d_short) - 1):
#     for j in range(no_intervals_periods):
#         p_d_long.append(p_d_short[i] + (p_d_short[i + 1] - p_d_short[i]) / no_intervals_periods * j)
#
# # i should be 46 at this time
# i = len(p_d_short) - 2  # make sure i is 46
# for j in range(no_intervals_periods):
#     p_d_long.append(p_d_short[i + 1] + (p_d_short[i + 1] - p_d_short[i]) / no_intervals_periods * j)
#
# p_d_min = p_d_long[0] - p_d_long[0] / 3
# p_d_max = p_d_long[-1]

# p_d = []
# with open('probability.csv') as f:
#     reader = csv.reader(f, delimiter=',', quotechar='|')
#     for row in reader:
#         p_d.append([int(float(i)) for i in row])

# prices2 = [141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141,
#            141,
#            141, 141, 141,
#            141, 141, 141, 141, 141, 143, 143, 143, 142, 142, 142, 148, 148, 148, 189, 189, 189, 163,
#            163,
#            163, 145, 145,
#            145, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141,
#            141,
#            141, 141, 141,
#            141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 142, 142, 142, 141, 141, 141,
#            146,
#            146, 146, 158,
#            158, 158, 203, 203, 203, 313, 313, 313, 1155, 1155, 1155, 1155, 1155, 1155, 1446, 1446,
#            1446,
#            616, 616, 616,
#            616, 616, 616, 363, 363, 363, 363, 363, 363, 313, 313, 313, 427, 427, 427, 221, 221, 221,
#            158,
#            158, 158, 148,
#            148, 148, 158, 158, 158, 143, 143, 143, 142, 142, 142, 144, 144, 144]

# seed = r.uniform(p_d_min, p_d_max)
# p_start = bisect.bisect(p_d_long, seed)
# middle_point = bisect.bisect(p_d_long, seed)
# p_start = (middle_point - int(duration / 2)) % no_intervals
# p_start = int(np.random.choice(period_options, 1, p=p_d_short)[0] * no_intervals_periods + r.randint(0, 3))

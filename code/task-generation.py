from numpy import genfromtxt
from numpy import sqrt, pi, random
import random as r
import numpy as np

no_intervals = 144
no_periods = 48
no_intervals_periods = int(no_intervals / no_periods)
care_f_weight = 9000
no_tasks = 5
max_demand_multiplier = no_tasks

p_d = genfromtxt('inputs/probability.csv', delimiter=',', dtype="float")
p_d_short = [int(p) for p in p_d[0]]
sum_t = sum(p_d_short)
p_d_short = [p / sum_t for p in p_d_short]

l_demands = genfromtxt('inputs/demands_list.csv', delimiter=',', dtype="float")

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
    demand = int(demand * 1000)
    demands.append(demand)

    # job duration
    duration = max(1, int(random.rayleigh(mode_value, 1)[0]))
    durations.append(duration)

    # job preferred start time
    p_start = no_intervals + 1
    while p_start + duration - 1 >= no_intervals - 1 or p_start < 0:
        middle_point = int(np.random.choice(a=no_periods, size=1, p=p_d_short)[0]
                           * no_intervals_periods
                           + np.random.random_integers(low=-2, high=2))
        p_start = middle_point - int(duration / 2)
    preferred_starts.append(p_start)

    # job earliest starting time
    e_start = 0
    earliest_starts.append(e_start)

    # job latest finish time
    l_finish = no_intervals - 1
    latest_ends.append(l_finish)

    # job care factor
    care_f = round(r.random(), 1)
    if care_f == 0:
        care_f = 0.01
    care_f = int(care_f * care_f_weight)
    care_factors.append(care_f)

    if r.choice([True, False]) and counter_j > 0:
        successors.append(counter_j + 1)

        id_predecessor = np.random.random_integers(low=1, high=counter_j)

        while durations[id_predecessor] + duration > no_intervals:
            id_predecessor = np.random.random_integers(low=1, high=counter_j)

        predecessors.append(int(id_predecessor))
        delay = 0 if durations[id_predecessor] + duration == no_intervals \
            else np.random.random_integers(low=0, high=no_intervals - durations[id_predecessor] - duration)
        prec_delays.append(int(delay))

    for d in range(duration):
        aggregated_loads[p_start + d] += demand

no_precedences = len(predecessors)
maximum_demand = max(demands) * max_demand_multiplier

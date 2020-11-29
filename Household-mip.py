# Version 2 combines scheduling and aggregating loads together
# Version 3 uses a for loop to find the cheapest time slot, instead of using the list comprehension


from time import time
from gurobipy import *

show_astart = True

prices = [141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141,
          141, 141, 141, 141, 141, 143, 143, 143, 142, 142, 142, 148, 148, 148, 189, 189, 189, 163, 163, 163, 145, 145,
          145, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141,
          141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 142, 142, 142, 141, 141, 141, 146, 146, 146, 158,
          158, 158, 203, 203, 203, 313, 313, 313, 1155, 1155, 1155, 1155, 1155, 1155, 1446, 1446, 1446, 616, 616, 616,
          616, 616, 616, 363, 363, 363, 363, 363, 363, 313, 313, 313, 427, 427, 427, 221, 221, 221, 158, 158, 158, 148,
          148, 148, 158, 158, 158, 143, 143, 143, 142, 142, 142, 144, 144, 144]
no_intervals = len(prices)
INTERVALS = range(no_intervals)

demands = [55, 15, 300, 700, 80, 2400, 3500, 1500, 400, 15]
durations = [3, 3, 1, 2, 3, 4, 3, 6, 3, 4]
preferred_starts = [41, 135, 96, 115, 6, 120, 71, 31, 73, 46]
earliest_starts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
latest_ends = [143, 143, 143, 143, 143, 143, 143, 143, 143, 143]
care_factors = [1, 0, 9, 2, 1, 4, 3, 2, 1, 7]

num_precedences = 2
predecessors = [1, 7]
successors = [9, 10]
prec_delays = [62, 127]

percent = 1

m = Model("household_load_scheduling")

devices = []
DEVICES = range(len(demands))
PREC = range(num_precedences)
max_demand = int(sum(demands) * percent)

run_costs = [
    [
        care_factors[d] * abs(s - preferred_starts[d])
        + sum(
            [
                prices[t] * demands[d]
                for t in range(s, min(s + durations[d], no_intervals))
            ]
        )
        for s in INTERVALS
    ]
    for d in DEVICES
]

for d in DEVICES:
    d_times = []
    for _ in INTERVALS:
        d_t = m.addVar(vtype=GRB.BINARY)
        d_times.append(d_t)
    devices.append(d_times)
    m.addConstr(sum(d_times) == 1)

# for _ in DEVICES:
#     d_t = m.addVar(INTERVALS, vtype=GRB.INTEGER)

for d in DEVICES:
    astart = sum([devices[d][t] * t for t in INTERVALS])
    m.addConstr(earliest_starts[d] <= astart)
    m.addConstr(astart + durations[d] - 1 <= latest_ends[d])

for t in INTERVALS:
    interval_demand = sum([devices[i][t] * demands[i] for i in DEVICES])
    m.addConstr(interval_demand <= max_demand)
#
for p in PREC:
    pre = predecessors[p] - 1
    succ = successors[p] - 1
    d = prec_delays[p]

    astart_pre = sum([devices[pre][t] * t for t in INTERVALS])
    astart_succ = sum([devices[succ][t] * t for t in INTERVALS])

    m.addConstr(astart_pre + durations[pre] <= astart_succ)
    m.addConstr(astart_succ <= astart_pre + durations[pre] + d)

m.setObjective(sum([sum([devices[i][t] * run_costs[i][t] for t in INTERVALS])
                    for i in DEVICES]), GRB.MINIMIZE)

m.setParam('OutputFlag', True)
m.optimize()

# print("g solve time", time() - start)
#
# print('\nTOTAL COSTS: %g' % m.objVal)
# print('SOLUTION:')

astarts = [sum([int(i * t.x) for i, t in enumerate(d)]) + 1
           for d in devices]

if show_astart:
    print("g solution", astarts, m.objVal)

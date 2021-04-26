from minizinc import *
from numpy import sqrt, pi, random
import numpy as np
import random as r

from pandas import DataFrame, IndexSlice, date_range, Series, concat
from numpy import genfromtxt
from time import strftime, localtime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

# modifiable parameters
num_households = 2
no_tasks = 5
no_days = 2
care_f_max = 10
# care_f_weight = 0
# care_f_weight = int(care_f_weight)
incon_weights = [
    # 0,
    # 5,
    # 20,
    # 50,
    # 80,
    # 200,
    # 500,
    800,
    # 1000
]
max_demand_multiplier = no_tasks
display_months = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
var_choices = {
    0: "input_order",
    1: "first_fail",
    2: "anti_first_fail",
    3: "smallest",
    4: "largest",
    # 5: "dom_w_deg",
    # 6: "impact",
    # 7: "max_regret",
    # 8: "most_constrained",
    # 9: "occurrence"
}
const_choices = {
    # 0: "indomain",
    # 1: "indomain_interval",
    2: "indomain_max",
    3: "indomain_median",
    # 4: "indomain_middle",
    5: "indomain_min",
    6: "indomain_random",
    # 7: "indomain_reverse_split",
    # 8: "indomain_split",
    # 9: "indomain_split_random",
    # 10: "outdomain_max",
    # 11: "outdomain_median",
    # 12: "outdomain_min",
    # 13: "outdomain_random"
}

# fixed parameters
no_intervals = 144
no_periods = 48
no_intervals_periods = int(no_intervals / no_periods)
current_time = strftime("%m%d-%H%M", localtime())
folder_name = strftime("%m%d/%H%M", localtime())
directory = 'results/' + folder_name
if num_households < 50:
    directory += '-dummy'
colour_choices = "Set1"


def read_data():
    prices = genfromtxt('inputs/prices.csv', delimiter=',', dtype="float").astype(int)  # in cents

    models = dict()
    models["Initial"] = dict()
    models["Initial"]["CP"] = 'scripts/Household-cp.mzn'
    # scripts["Initial"]["MIP"] = 'scripts/Household-mip.mzn'
    models["Modified"] = dict()
    models["Modified"]["CP"] = 'scripts/Household-cp-pre.mzn'
    # scripts["Modified"]["MIP"] = 'scripts/Household-mip-pre.mzn'

    solvers = dict()
    solvers["MIP"] = ["Gurobi"]
    solvers["CP"] = ["Gecode"]
    # solvers["CP"] = ["Chuffed"]
    # solvers["MIP"] = ["Gurobi", "Cplex"]
    # solvers["CP"] = ["Gecode", "ORtools"]

    return prices, models, solvers


def task_generation():
    p_d = genfromtxt('inputs/probability.csv', delimiter=',', dtype="float")
    p_d_short = [int(p) for p in p_d[0]]
    sum_t = sum(p_d_short)
    p_d_short = [p / sum_t for p in p_d_short]

    l_demands = genfromtxt('inputs/demands_list.csv', delimiter=',', dtype="float")

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
        demand = int(demand * 1000)
        demands.append(demand)

        # job duration
        duration = max(1, int(random.rayleigh(mode_value, 1)[0]))
        # while duration == 0:
        #     duration = int(random.rayleigh(mode_value, 1)[0])
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
        # e_start = r.randint(0, max(p_start - 1, 0))
        earliest_starts.append(e_start)

        # job latest finish time
        l_finish = no_intervals - 1
        # l_finish = r.randint(p_start + duration, min(no_intervals - 1, p_start + duration))
        latest_ends.append(l_finish)

        # job care factor
        care_f = int(r.choice([i for i in range(care_f_max + 1)]))
        care_factors.append(care_f)

        if r.choice([True, False]) and counter_j > 0:

            # task predecessor
            id_predecessor_set = [i for i in range(counter_j)]
            id_predecessor = r.choice(id_predecessor_set)

            while preferred_starts[id_predecessor] + durations[id_predecessor] - 1 >= preferred_starts[counter_j] \
                    and len(id_predecessor_set) > 0:
                id_predecessor_set.remove(id_predecessor)
                if len(id_predecessor_set) > 0:
                    id_predecessor = r.choice(id_predecessor_set)

            if len(id_predecessor_set) > 0:
                predecessors.append(int(id_predecessor))
                successors.append(counter_j)

                # predecessing delay
                delay = 0
                if not durations[id_predecessor] + duration - 1 == no_intervals - 1:
                    delay = r.choice([i for i in range(no_intervals + 1 - duration - durations[id_predecessor])])
                prec_delays.append(int(delay))

        for d in range(duration):
            aggregated_loads[p_start + d] += demand

    no_precedences = len(predecessors)
    maximum_demand = max(demands) * max_demand_multiplier

    print(" --- Household made ---")

    return no_intervals, preferred_starts, no_tasks, earliest_starts, latest_ends, durations, demands, care_factors, \
           no_precedences, predecessors, successors, prec_delays, maximum_demand, aggregated_loads


def data_preprocessing(prices_input, earliest_starts, latest_ends, durations, preferred_starts, care_factors, demands,
                       care_f_weight):
    run_costs = []
    for i in range(no_tasks):
        run_cost_task = []
        for t in range(no_intervals):
            if earliest_starts[i] <= t <= latest_ends[i] - durations[i] + 1:
                rc = abs(t - preferred_starts[i]) * care_factors[i] * care_f_weight \
                     + sum([prices_input[t2] for t2 in range(t, t + durations[i])]) * demands[i]
            else:
                rc = (no_intervals * care_f_max * care_f_weight + max(prices_input) * max(demands) * max(durations))
            run_cost_task.append(int(rc))
        run_costs.append(run_cost_task)
    return run_costs


def optimistic_search(num_intervals, num_tasks, durations, demands, predecessors, successors,
                      prec_delays, max_demand, run_costs, preferred_starts, latest_ends):
    start_time = timeit.default_timer()

    solutions = []
    household_profile = [0] * num_intervals
    obj = 0

    for i in range(num_tasks):

        task_demand = demands[i]
        task_duration = durations[i]

        max_cost = max(run_costs[i])
        feasible_indices = [t for t, x in enumerate(run_costs[i]) if x < max_cost]

        if i in successors:
            pos_in_successors = successors.index(i)
            pre_id = predecessors[pos_in_successors]
            pre_delay = prec_delays[pos_in_successors]
            this_aestart = solutions[pre_id] + durations[pre_id]
            feasible_indices = [f for f in feasible_indices if this_aestart <= f <= this_aestart + pre_delay]

        if i in predecessors:
            indices_in_precs = [i for i, k in enumerate(predecessors) if k == i]
            for iip in indices_in_precs:
                suc_id = successors[iip]
                suc_duration = durations[suc_id]
                suc_lend = latest_ends[suc_id]
                feasible_indices = [f for f in feasible_indices if f + task_duration + suc_duration - 1 <= suc_lend]

        # if not this_successors == []:
        #     last_suc_lend = latest_ends[max(this_successors)]
        #     sum_successors_durations = sum([durations[t] for t in this_successors])
        #     for j in feasible_indices:
        #         if not (j + task_duration + sum_successors_durations - 1 <= last_suc_lend):
        #             feasible_indices.remove(j)

        # last_prec_delay = -1
        # this_predecessors = []
        # while this_id in successors:

        # if not this_predecessors == []:
        #     first_prec_astart = solutions[min(this_predecessors)]
        #     sum_prec_durations = sum([durations[t] for t in this_predecessors])
        #     this_estart = first_prec_astart + sum_prec_durations
        #     for j in feasible_indices:
        #         if not (this_estart <= j <= this_estart + last_prec_delay):
        #             feasible_indices.remove(j)

        if feasible_indices is []:
            print("error")

        feasible_min_cost = min([run_costs[i][f] for f in feasible_indices])
        feasible_min_cost_indices = [f for f in feasible_indices if run_costs[i][f] == feasible_min_cost]
        a_start = r.choice(feasible_min_cost_indices)

        # check max demand constraint
        max_demand_starts = dict()
        temp_profile = household_profile[:]
        try:
            for d in range(a_start, a_start + task_duration):
                temp_profile[d] += task_demand
        except:
            print("error")
        temp_max_demand = max(temp_profile)
        while temp_max_demand > max_demand and len(feasible_indices) > 1:

            max_demand_starts[a_start] = temp_max_demand
            feasible_indices.remove(a_start)

            feasible_min_cost = min([run_costs[i][f] for f in feasible_indices])
            feasible_min_cost_indices = [k for k, x in enumerate(run_costs[i]) if x == feasible_min_cost]
            a_start = r.choice(feasible_min_cost_indices)

            temp_profile = household_profile[:]
            for d in range(a_start, a_start + task_duration):
                temp_profile[d] += task_demand
            temp_max_demand = max(temp_profile)

        if len(feasible_indices) == 0 and not max_demand_starts:
            a_start = min(max_demand_starts, key=max_demand_starts.get)

        solutions.append(a_start)
        for d in range(a_start, a_start + task_duration):
            household_profile[d] += task_demand
        obj += run_costs[i][a_start]

    elapsed = timeit.default_timer() - start_time

    return solutions, household_profile, obj, elapsed


def optimal_solving(model_file, num_intervals, prices_day, preferred_starts, num_tasks, earliest_starts, latest_ends,
                    durations, demands, care_factors, num_precedences, predecessors, successors, prec_delays,
                    max_demand, solver_choice, preprocessing_choice, run_costs, s_type, search, care_f_weight):
    sing_dsp = Instance([model_file])
    sing_dsp["num_intervals"] = num_intervals
    sing_dsp["num_tasks"] = num_tasks
    sing_dsp["durations"] = durations
    sing_dsp["demands"] = demands
    sing_dsp["num_precedences"] = num_precedences
    sing_dsp["predecessors"] = [p + 1 for p in predecessors]
    sing_dsp["successors"] = [s + 1 for s in successors]
    sing_dsp["prec_delays"] = prec_delays
    sing_dsp["max_demand"] = max_demand

    if "ini" in preprocessing_choice.lower():
        sing_dsp["prices"] = prices_day
        sing_dsp["preferred_starts"] = [ps + 1 for ps in preferred_starts]
        sing_dsp["earliest_starts"] = [es + 1 for es in earliest_starts]
        sing_dsp["latest_ends"] = [le + 1 for le in latest_ends]
        sing_dsp["care_factors"] = [cf * care_f_weight for cf in care_factors]
    else:
        sing_dsp["run_costs"] = run_costs

    sing_dsp.add_to_model("solve ")
    if solver_choice == "Gecode":
        sing_dsp.add_to_model(":: {} ".format(search))
    sing_dsp.add_to_model("minimize obj;")

    solver = load_solver(solver_choice.lower())

    # if solver_choice == "Chuffed" or "ORtools":
    #     result = solver.solve(sing_dsp, tags=[s_type], free_search=True)
    # else:
    #     result = solver.solve(sing_dsp, tags=[s_type])
    #
    result = solver.solve(sing_dsp, tags=[s_type])
    # result = solver.solve(sing_dsp)

    # print(solver_choice)

    return result


def sing_dsp_experiments(prices, models, solvers):
    objs = dict()
    times = dict()
    households = dict()
    tasks = dict()
    num_days = min(no_days, len(prices))

    print("**********")
    print("{} days, {} houses, {} tasks".format(num_days, num_households, no_tasks))

    for h in range(num_households):

        print("===== house {} =====".format(h + 1))

        num_intervals, preferred_starts, num_tasks, earliest_starts, latest_ends, durations, demands, care_factors, \
        num_precedences, predecessors, successors, prec_delays, max_demand, household_profile = task_generation()

        for care_f_weight in incon_weights:

            tasks[0, h + 1, "demands"] = demands
            tasks[0, h + 1, "durations"] = durations
            tasks[0, h + 1, "earliest_starts"] = earliest_starts
            tasks[0, h + 1, "latest_ends"] = latest_ends
            tasks[0, h + 1, "care_factors"] = [cf * care_f_weight for cf in care_factors]
            tasks[0, h + 1, "predecessors"] = predecessors + [-1] * (num_tasks - num_precedences)
            tasks[0, h + 1, "successors"] = successors + [-1] * (num_tasks - num_precedences)
            tasks[0, h + 1, "prec_delays"] = prec_delays + [-1] * (num_tasks - num_precedences)
            tasks[0, h + 1, "preferred_starts"] = preferred_starts

            for day in range(num_days):

                # print("{}".format(day + 1))

                prices_day = prices[day]
                if np.isnan(prices_day[-1]) or len(prices_day) == no_periods:
                    prices_day = [int(p) for p in prices_day[:no_periods] for _ in range(no_intervals_periods)]
                else:
                    prices_day = [int(p) for p in prices_day]

                run_costs = data_preprocessing(prices_day, earliest_starts, latest_ends, durations, preferred_starts,
                                               care_factors, demands, care_f_weight)

                # preferred
                house_key = (day + 1, h + 1)
                households[house_key] = dict()
                households[house_key]["profile", "Preferred"] = household_profile
                households[house_key]["max", "Preferred"] = max(household_profile)
                households[house_key]["max", "Limit"] = max_demand
                obj_preferred = 0
                for i, p_start in zip(range(num_tasks), preferred_starts):
                    obj_preferred += run_costs[i][p_start]

                # heuristics
                sol_ogsa, optimistic_d_profile, obj_ogsa, run_time_ogsa \
                    = optimistic_search(num_intervals, num_tasks, durations, demands, predecessors, successors,
                                        prec_delays, max_demand, run_costs[:], preferred_starts, latest_ends)
                if ("profile", "ogsa") not in households[house_key]:
                    households[house_key]["profile", "OGSA"] = optimistic_d_profile
                    households[house_key]["max", "OGSA"] = max(optimistic_d_profile)

                    tasks[day + 1, h + 1, "astart - ogsa"] = sol_ogsa

                # solvers
                for m_type in models:  # ini or modified
                    for s_type in models[m_type]:  # mip or cp
                        for solver_choice in solvers[s_type]:

                            this_key = (solver_choice, m_type)
                            if this_key not in times and this_key not in objs:
                                times[this_key] = dict()
                                objs[this_key] = dict()

                            model_file = models[m_type][s_type]

                            run_time = -999
                            obj = 0
                            sol = [0] * no_intervals
                            for vc in var_choices:
                                for cc in const_choices:

                                    search = "int_search(actual_starts, {}, {}, complete)".format(var_choices[vc],
                                                                                                  const_choices[cc])
                                    solver_results = optimal_solving(model_file, num_intervals, prices_day,
                                                                     preferred_starts,
                                                                     num_tasks, earliest_starts, latest_ends, durations,
                                                                     demands,
                                                                     care_factors, num_precedences, predecessors,
                                                                     successors,
                                                                     prec_delays, max_demand, solver_choice, m_type,
                                                                     run_costs[:],
                                                                     s_type, search, care_f_weight)
                                    try:
                                        sol = solver_results._solutions[-1].assignments
                                        obj = solver_results._solutions[-1].objective
                                        run_time = solver_results._solutions[-1].statistics['time'].microseconds / 1000
                                    except:
                                        print("error")

                                    times[this_key][
                                        day + 1, h + 1, var_choices[vc], const_choices[cc], care_f_weight] \
                                        = run_time
                                    objs[this_key][
                                        day + 1, h + 1, var_choices[vc], const_choices[cc], care_f_weight] = obj
                                    if obj > obj_ogsa:
                                        print('error')

                                optimised_d_profile, actual_starts_op = optimised_profile(sol, s_type, demands,
                                                                                          durations)
                            households[house_key]["profile", solver_choice] = optimised_d_profile
                            households[house_key]["max", solver_choice] = max(optimised_d_profile)
                            tasks[day + 1, h + 1, "astart - {}".format(solver_choice)] = actual_starts_op

                # heuristic results
                optimistic_key = ("OGSA", m_type)
                if optimistic_key not in times:
                    times[optimistic_key] = dict()
                    objs[optimistic_key] = dict()
                times[optimistic_key][
                    day + 1, h + 1, var_choices[vc], const_choices[cc], care_f_weight] = run_time_ogsa
                objs[optimistic_key][
                    day + 1, h + 1, var_choices[vc], const_choices[cc], care_f_weight] = obj_ogsa

                # preferred results
                preferred_key = ("Preferred", m_type)
                if preferred_key not in objs:
                    objs[preferred_key] = dict()
                objs[preferred_key][
                    day + 1, h + 1, var_choices[vc], const_choices[cc], care_f_weight] = obj_preferred

    pd_times = DataFrame.from_dict(times)
    pd_times.index.names = ["Month", "House", "Variable choices", "Constraint variables", "Weight"]
    pd_objs = DataFrame.from_dict(objs)
    pd_objs.index.names = ["Month", "House", "Variable choices", "Constraint variables", "Weight"]
    pd_households = DataFrame.from_dict(households).transpose()
    pd_households.index.names = ["Month", "House"]
    pd_tasks = DataFrame.from_dict(tasks).transpose()
    pd_tasks.index.names = ["Month", "House", "Attribute"]

    print("----------")
    print("###### The experiment is finished. ######")

    return pd_households, pd_times, pd_objs, pd_tasks


def optimised_profile(solutions, s_type, demands, durations):
    solutions = list(solutions.values())[0]

    if s_type.lower() == "mip":
        actual_starts = [sum([i * int(v) for i, v in enumerate(row)]) for row in solutions]
    else:
        # need to change the index back to starting from 0!!!!!
        actual_starts = [int(a) - 1 for a in solutions]

    optimised_demand_profile = [0] * no_intervals
    for demand, duration, a_start, i in zip(demands, durations, actual_starts, range(no_tasks)):
        for t in range(a_start, a_start + duration):
            optimised_demand_profile[t] += demand

    return optimised_demand_profile, actual_starts


def write_results(pd_households, pd_times, pd_objs, pd_tasks, pd_time_mma):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # household demand profiles
    house_file_name = directory + '/demand-house-{}.csv'.format(current_time)
    pd_demand = pd_households.stack()
    pd_demand.index.names = ["Month", "House", "Demand profile"]
    pd_demand.columns = ["Max demand (W)", "Demand (W)"]
    pd_demands_profiles_only = pd_demand["Demand (W)"].apply(Series)
    pd_demands_profiles_columns = date_range("00:00", periods=no_intervals,
                                             freq="{}min".format(int(60 * 24 / no_intervals)))
    pd_demands_profiles_only.columns = [d.strftime("%H:%M") for d in pd_demands_profiles_columns]
    pd_demand = concat([pd_demand[:], pd_demands_profiles_only[:]], axis=1).drop("Demand (W)", axis=1)
    pd_demand.to_csv(house_file_name, sep=',', encoding='utf-8')

    # max demand
    max_demand_file_name = directory + '/demand-max-{}.csv'.format(current_time)
    pd_max_demands = pd_households["max"]
    pd_max_demands.to_csv(max_demand_file_name, sep=',', encoding='utf-8')

    # obj and time
    out_file_name = directory + '/obj-{}.csv'.format(current_time)
    pd_objs.to_csv(out_file_name, sep=',', encoding='utf-8')

    out_file_name = directory + '/time-all-{}.csv'.format(current_time)
    pd_times.to_csv(out_file_name, sep=',', encoding='utf-8')

    out_file_name = directory + '/time-mma-{}.csv'.format(current_time)
    pd_time_mma.to_csv(out_file_name, sep=',', encoding='utf-8')

    # household tasks
    out_file_name = directory + '/tasks-{}.csv'.format(current_time)
    pd_tasks.to_csv(out_file_name, sep=',', encoding='utf-8')

    out_file_name = directory + '/parameters-{}.txt'.format(current_time)
    para_str = "care factor weights = {}\n" \
               "num of tasks = {}\n" \
               "max demand mutiplier = {}\n".format(incon_weights, no_tasks, max_demand_multiplier)
    f = open(out_file_name, "w")
    f.write(para_str)
    f.close()

    print("###### Results are written. ######")


def plot_results(pd_households, pd_times, pd_objs, prices, models):
    if not os.path.exists(directory):
        os.makedirs(directory)

    sns.set(style="darkgrid")
    sns.set_context("paper", font_scale=1.5)
    num_prices = len(prices)

    # draw objective plot
    pd_obj = pd_objs.stack().stack().reset_index()
    pd_obj.columns = ["Month", "House", "Variable choices", "Constraint variables", "Weight", "Model", "Solver",
                      "Objective"]
    plot_obj = sns.relplot(x="House",
                           y="Objective",
                           col="Month",
                           col_wrap=3,
                           hue="Solver",
                           data=pd_obj,
                           aspect=1,
                           palette=colour_choices,
                           # alpha=.75,
                           s=60)
    plot_obj_file_name = directory + "/obj-{}.png".format(current_time)
    plot_obj.savefig(plot_obj_file_name)
    plt.clf()

    # draw obj ratio plot
    # pd_obj_ratio = pd_objs
    if ("Gecode", "Modified") in pd_objs and ("OGSA", "Modified") in pd_objs:
        pd_objs[("Objective Ratio", "Modified")] \
            = pd_objs[("OGSA", "Modified")] / pd_objs[("Gecode", "Modified")]

    if ("Gecode", "Initial") in pd_objs and ("OGSA", "Initial") in pd_objs:
        pd_objs[("Objective Ratio", "Initial")] \
            = pd_objs[("OGSA", "Initial")] / pd_objs[("Gecode", "Initial")]

    pd_obj_ratio = pd_objs["Objective Ratio"].stack().reset_index()
    pd_obj_ratio.columns = ["Month", "House", "Variable choices", "Constraint variables", "Weight", "Model",
                            "Objective Ratio"]
    plot_obj_ratio = sns.relplot(x="House",
                                 y="Objective Ratio",
                                 col="Month",
                                 col_wrap=3,
                                 # hue="Solver",
                                 data=pd_obj_ratio,
                                 aspect=1,
                                 palette=colour_choices,
                                 # alpha=.75,
                                 s=60)
    plot_obj_ratio.map(plt.axhline, y=1, ls=":", c=".5")
    plot_obj_ratio_file_name = directory + "/ratio-obj-{}.png".format(current_time)
    plot_obj_ratio.savefig(plot_obj_ratio_file_name)
    plt.clf()

    # draw time plot
    pd_time = pd_times
    pd_time = pd_time.stack().stack().reset_index()
    pd_time.columns = ["Month", "House", "Variable choices", "Constraint variables", "Weight", "Model", "Solver",
                       "Time (ms)"]
    plot_time = sns.relplot(x="House",
                            y="Time (ms)",
                            col="Month",
                            col_wrap=3,
                            hue="Solver",
                            style="Model",
                            data=pd_time,
                            aspect=1,
                            palette=colour_choices,
                            alpha=.75,
                            s=60)
    plot_time_file_name = directory + "/time-all-{}.png".format(current_time)
    plot_time.savefig(plot_time_file_name)
    plt.clf()

    for m in range(no_days):
        pd_time = pd_times.xs(m + 1, level="Month")
        pd_time = pd_time.stack().stack().reset_index()
        pd_time.columns = ["House", "Select", "Choice", "Weight", "Model", "Solver", "Time (ms)"]
        plot_time = sns.FacetGrid(pd_time,
                                  col="Select",
                                  row="Choice",
                                  hue="Model",
                                  palette=colour_choices,
                                  aspect=1,
                                  margin_titles=True,
                                  legend_out=True,
                                  # size=60,
                                  # alpha=.75,
                                  )
        plot_time = plot_time.map_dataframe(plt.scatter,
                                            "House",
                                            "Time (ms)",
                                            # style="Model",
                                            )
        # plot_time.add_legend()
        plot_time_file_name = directory + "/time-{}-{}.png".format(m + 1, current_time)
        plot_time.savefig(plot_time_file_name)
    plt.clf()

    # draw time ratio plot
    pd_ratio = pd_times

    if ("Gurobi", "Initial") in pd_ratio and ("Gecode", "Initial") in pd_ratio:
        pd_ratio["Solver Run Time Ratio", "Initial"] \
            = pd_ratio[("Gurobi", "Initial")] / pd_ratio[("Gecode", "Initial")]
    if ("Gurobi", "Modified") in pd_ratio and ("Gecode", "Modified") in pd_ratio:
        pd_ratio["Solver Run Time Ratio", "Modified"] \
            = pd_ratio[("Gurobi", "Modified")] / pd_ratio[("Gecode", "Modified")]
    if ("Gurobi", "Initial") in pd_ratio and ("Gurobi", "Modified") in pd_ratio:
        pd_ratio[("Model Run Time Ratio", "Gurobi")] \
            = pd_ratio[("Gurobi", "Initial")] / pd_ratio[("Gurobi", "Modified")]
    if ("Gecode", "Initial") in pd_ratio and ("Gecode", "Modified") in pd_ratio:
        pd_ratio[("Model Run Time Ratio", "Gecode")] \
            = pd_ratio[("Gecode", "Initial")] / pd_ratio["Gecode", "Modified"]

    ratios = []
    if "Solver Run Time Ratio" in pd_ratio:
        ratios.append("Solver")
    if "Model Run Time Ratio" in pd_ratio:
        ratios.append("Model")

    for ratio_type in ratios:
        c_name = "Model" if ratio_type == "Solver" else "Solver"
        c_name2 = "{} Run Time Ratio".format(ratio_type)
        pd_ratio_type = pd_ratio[c_name2].stack().reset_index()
        pd_ratio_type.columns = ["Month", "House", "Variable choices", "Constraint variables", "Weight", c_name,
                                 c_name2]
        plot_ratio_type = sns.relplot(x="House",
                                      y=c_name2,
                                      col="Month",
                                      col_wrap=3,
                                      hue=c_name,
                                      data=pd_ratio_type,
                                      aspect=1,
                                      palette=colour_choices,
                                      alpha=.75,
                                      s=60)
        # plot_ratio_type.set_yticklabels(list(range(int(min(np.ceil(max(pd_ratio_type[c_name2])), 10)))))
        plot_ratio_type.map(plt.axhline, y=1, ls=":", c=".5")
        plot_ratio_type_file_name = directory + "/ratio-{}-{}.png".format(ratio_type.lower(), current_time)
        plot_ratio_type.savefig(plot_ratio_type_file_name)
        plt.clf()

    # draw max-min-avg plot
    pd_time_mma = pd_times.groupby("Weight").agg(["min", "max", "mean"]).loc[:,
                  (["Gurobi", "Gecode", "OGSA"], slice(None), slice(None))].stack().stack().stack().reset_index()
    pd_time_mma.columns = ["Weight", "Type", "Model", "Solver", "Time (ms)"]
    plot_time_mma = sns.relplot(x="Weight",
                                y="Time (ms)",
                                hue="Type",
                                style="Model",
                                col="Solver",
                                col_wrap=3,
                                data=pd_time_mma,
                                aspect=1,
                                linewidth=2,
                                kind="line",
                                palette=colour_choices,
                                markers=["o", "v"],
                                markersize=8,
                                alpha=.7,
                                # s=60
                                )
    # plot_time_mma.set_xticklabels(list(incon_weights))
    plot_time_mma_file_name = directory + "/time-mma-{}.png".format(current_time)
    plot_time_mma.savefig(plot_time_mma_file_name)
    plt.clf()

    # draw demand plot
    idx = IndexSlice
    display_months2 = display_months[:min(no_days, num_prices)]
    display_houses = np.random.choice(num_households, size=min(12, num_households), replace=False).tolist()
    display_houses.sort()
    pd_demand = pd_households.loc[idx[display_months2, display_houses], :].stack()
    pd_demand.index.names = ["Month", "House", "Demand profile"]
    pd_demand.columns = ["Max Demand (W)", "Demand (W)"]
    pd_demand_profiles = pd_demand["Demand (W)"]
    pd_demands_profiles_only = pd_demand_profiles.apply(Series)
    pd_demands_profiles_columns = date_range("00:00", periods=no_intervals,
                                             freq="{}min".format(int(60 * 24 / no_intervals)))
    pd_demands_profiles_only.columns = [d.strftime("%H:%M") for d in pd_demands_profiles_columns]
    pd_demand_profiles = concat([pd_demand_profiles[:], pd_demands_profiles_only[:]], axis=1).drop("Demand (W)", axis=1)
    pd_demand_profiles = pd_demand_profiles.stack().reset_index()
    pd_demand_profiles.columns = ["Month", "House", "Demand profile", "Time", "Demand (W)"]
    plot_demand = sns.relplot(x="Time",
                              y="Demand (W)",
                              hue="Month",
                              style="Demand profile",
                              col="House",
                              col_wrap=3,
                              data=pd_demand_profiles,
                              aspect=1.5,
                              linewidth=3,
                              kind="line",
                              palette=colour_choices)
    plot_demand.set(xticks=[0, 12 * 3, 24 * 3, 36 * 3, 0])
    plot_demand_file_name = directory + "/demand-{}.png".format(current_time)
    plot_demand.savefig(plot_demand_file_name)
    plt.clf()

    # draw max demand plot
    pd_max_demand = pd_households.stack().reset_index()
    pd_max_demand.columns = ["Month", "House", "Demand profile", "Max demand (W)", "Demand (W)"]
    plot_max_demand = sns.relplot(x="House",
                                  y="Max demand (W)",
                                  col="Month",
                                  col_wrap=3,
                                  hue="Demand profile",
                                  data=pd_max_demand,
                                  aspect=1,
                                  palette=colour_choices,
                                  # alpha=.75,
                                  s=60)
    plot_mdemand_file_name = directory + "/demand-max-{}.png".format(current_time)
    plot_max_demand.savefig(plot_mdemand_file_name)
    plt.clf()

    # draw max demand ratio plot
    # pd_obj_ratio = pd_results.xs("obj", level="Result")
    # if ("Gecode", "Modified") in pd_results and ("OGSA", "Modified") in pd_results:
    #     pd_obj_ratio[("Objective Ratio", "Modified")] \
    #         = pd_obj_ratio[("OGSA", "Modified")] / pd_obj_ratio[("Gecode", "Modified")]
    #
    # if ("Gecode", "Initial") in pd_results and ("OGSA", "Initial") in pd_results:
    #     pd_obj_ratio[("Objective Ratio", "Initial")] \
    #         = pd_obj_ratio[("OGSA", "Initial")] / pd_obj_ratio[("Gecode", "Initial")]
    #
    # pd_obj_ratio = pd_obj_ratio.stack().reset_index()
    # pd_obj_ratio.columns = ["Month", "House", "Variable choices", "Constraint variables", "Model", "", "",
    #                         "Objective Ratio"]
    # plot_obj_ratio = sns.relplot(x="House",
    #                              y="Objective Ratio",
    #                              col="Month",
    #                              col_wrap=3,
    #                              # hue="Solver",
    #                              data=pd_obj_ratio,
    #                              aspect=1,
    #                              palette="Set2",
    #                              alpha=.75,
    #                              s=60)
    # plot_obj_ratio_file_name = directory + "/ratio-obj-{}.png".format(current_time)
    # plot_obj_ratio.savefig(plot_obj_ratio_file_name)

    # draw price plot
    # pd_prices_columns = Series([Timedelta(minutes=10 * i) for i in range(no_periods)])
    # pd_prices_columns = [str(d).split(" ")[-1][:-2] for d in pd_prices_columns]

    pd_prices_columns = date_range(start="00:00", periods=no_periods,
                                   freq="{}min".format(int(60 * 24 / no_periods)))
    pd_prices_columns = [d.strftime("%H:%M") for d in pd_prices_columns]
    pd_prices = DataFrame(data=prices[:num_prices], index=range(1, num_prices + 1), columns=pd_prices_columns)
    pd_prices = pd_prices.stack().reset_index()
    pd_prices.columns = ["Month", "Time", "Price"]
    plot_prices = sns.relplot(x="Time",
                              y="Price",
                              col="Month",
                              col_wrap=3,
                              data=pd_prices,
                              height=3,
                              aspect=1.5,
                              linewidth=3,
                              kind="line")
    plot_prices.set(xticks=[0, 12, 24, 36, 0])
    plot_prices_file_name = directory + "/prices-{}.png".format(current_time)
    plot_prices.savefig(plot_prices_file_name)
    plt.clf()

    print("###### Plots are drawn. ######")

    return pd_time_mma


def main():
    prices, models, solvers = read_data()
    pd_households, pd_times, pd_objs, pd_tasks = sing_dsp_experiments(prices, models, solvers)
    pd_time_mma = plot_results(pd_households, pd_times, pd_objs, prices, models)
    write_results(pd_households, pd_times, pd_objs, pd_tasks, pd_time_mma)

    print("###### All done! ######")


main()

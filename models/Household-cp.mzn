include "globals.mzn";

% parameters
int: num_intervals;
set of int: INTERVALS = 1..num_intervals; 

array[INTERVALS] of int: prices;

int: num_tasks;
set of int: TASKS = 1..num_tasks;
array[TASKS] of int: preferred_starts;
array[TASKS] of int: earliest_starts;
array[TASKS] of int: latest_ends; 
array[TASKS] of int: durations;
array[TASKS] of int: demands;
array[TASKS] of int: care_factors;

int: num_precedences;
set of int: PREC = 1 .. num_precedences;
array[PREC] of TASKS: predecessors;
array[PREC] of TASKS: successors;
array[PREC] of int: prec_delays;

% int: max_demand = ceil(max(demands) * 1.1);
int: max_demand;


% variables
array[TASKS] of var INTERVALS: actual_starts;
array [TASKS, INTERVALS] of int: run_costs = array2d(TASKS,INTERVALS, [ 
care_factors[d] * abs(s - preferred_starts[d]) + sum (t in s..min(s + durations[d] - 1, num_intervals)) (prices[t] * demands[d]) 
| d in TASKS, s in INTERVALS]);

% objectives
var int: obj= sum (d in TASKS) (run_costs[d, actual_starts[d]]);

constraint forall (d in TASKS) (earliest_starts[d] <= actual_starts[d] /\ actual_starts[d] + durations[d] - 1 <= latest_ends[d]);

constraint forall (p in PREC) (
    let { TASKS: pre = predecessors[p] ;
           TASKS: succ = successors[p] ;
           int: d = prec_delays[p]; } in
     actual_starts[pre] + durations[pre] <= actual_starts[succ] /\ actual_starts[succ] <= actual_starts[pre] + durations[pre] + d
   ); 
   
constraint cumulative(actual_starts, durations, demands, max_demand);

% solve 
% :: int_search(actual_starts, first_fail, indomain_max, complete)
% :: int_search(actual_starts, dom_w_deg, most_constrained, complete)
% minimize obj;

% output [show(actual_starts) ++ "\n" ++ show(obj)];


% array[INTERVALS] of float: prices;

% objectives
% array [TASKS, INTERVALS] of int: run_price = array2d(TASKS,INTERVALS, [ sum (t in s..min(s+durations[d], no_intervals-1)) (prices[t]) | d in TASKS, s in INTERVALS]);

% array [TASKS, INTERVALS] of float: run_price = array2d(TASKS,INTERVALS, [ care_factors[d]*abs(s - preferred_starts[d]) + sum (t in s..min(s+durations[d], no_intervals-1)) (prices[t]) | d in TASKS, s in INTERVALS]);

%array [TASKS, 1..2] of var 0..no_intervals: offset;
%constraint forall (d in TASKS) (offset[d,1] - offset[d, 2] = actual_starts[d] - preferred_starts[d]);

%solve minimize sum (d in TASKS) (run_price[d, actual_starts[d]] +  (offset[d,1] + offset[d,2]) * care_factors[d] ); 
% solve minimize obj = sum (d in TASKS) (run_price[d, actual_starts[d]]);

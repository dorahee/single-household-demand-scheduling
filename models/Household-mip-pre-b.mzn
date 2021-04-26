
% parameters
int: num_intervals;
set of int: INTERVALS = 1..num_intervals; 

int: num_tasks;
set of int: TASKS = 1..num_tasks;
array[TASKS] of int: durations;
array[TASKS] of int: demands;

array [TASKS, INTERVALS] of int: run_costs;

int: num_precedences;
set of int: PREC = 1..num_precedences;
array[PREC] of TASKS: predecessors;
array[PREC] of TASKS: successors;
array[PREC] of int: prec_delays;
int: max_demand;
% int: max_demand = ceil(max(demands) * 1.1);

% battery related
int: capacity;
int: power;
int: init_energy;

% variables
% array[TASKS] of var INTERVALS: actual_starts;
array[INTERVALS] of var float: battery_profile;
array[INTERVALS] of var float: battery_soc;
array[TASKS, INTERVALS] of var 0..1: actual_starts;


% objective
% var int: obj= sum (d in TASKS) (run_costs[d, actual_starts[d]]);
var int: obj= sum (d in TASKS, s in INTERVALS) (run_costs[d, s] * actual_starts[d, s]);

% battery power rate constraint
constraint forall (i in INTERVALS) (
    battery_profile[i] <= power /\ battery_profile[i] >= -power
);

% battery capacity constraint
constraint forall (i in INTERVALS) (
    battery_soc[i] <= capacity /\ battery_soc[i] >= 0
);

% battery soc constraint
constraint battery_soc[0] = init_energy;
constraint battery_soc[num_intervals] = init_energy;
constraint forall (i in 1..num_intervals) (
    battery_soc[i] = battery_soc[i - 1] + battery_profile[i]
);


% job must run at one time constraint
constraint forall (d in TASKS) (
    sum(s in INTERVALS) (actual_starts[d, s]) == 1
);


% job predecessor constraint
constraint forall (p in PREC) (
    let { TASKS: pre = predecessors[p] ;
           TASKS: succ = successors[p] ;
           int: d = prec_delays[p]; 
%            INTERVALS: astart_pre = sum(s in INTERVALS) (actual_starts[pre,s] * s);
%            INTERVALS: astart_succ = sum(s in INTERVALS) (actual_starts[succ,s] * s);
           } in
     sum(s in INTERVALS) (actual_starts[pre,s] * s) + durations[pre] <= sum(s in INTERVALS) (actual_starts[succ,s] * s) /\ 
     sum(s in INTERVALS) (actual_starts[succ,s] * s) <= sum(s in INTERVALS) (actual_starts[pre,s] * s) + durations[pre] + d
     ); 

% resource capacity constraint   
constraint forall (s in INTERVALS) (
    sum(d in TASKS) (actual_starts[d, s] * demands[d]) <= max_demand
);


% solve minimize obj;

% output [show(actual_starts) ++ "\n" ++ show(obj)];


% array [TASKS, INTERVALS] of float: run_costs;

% solve minimize sum (d in TASKS) (run_costs[d, actual_starts[d]]);

% constraint forall (d in TASKS) (earliest_starts[d] <= actual_starts[d] /\ actual_starts[d] + durations[d] - 1 <= latest_ends[d]);

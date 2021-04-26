% parameters
int: num_intervals;
set of int: INTERVALS = 1..num_intervals; 
float: num_intervals_hour = num_intervals / 24;

% load profile
array[INTERVALS] of float: loads;

% price related
array[INTERVALS] of int: prices;

% battery related
int: max_energy;  % in kwh
float: power;  % in kw
float: energy_interval = power / num_intervals_hour;
int: init_energy;
float: max_demand;

% variables
array[INTERVALS] of var float: battery_soc;  % soc is the energy in kwh

% decisions
array[INTERVALS] of var float: battery_charge;  % charge in kw
array[INTERVALS] of var float: battery_discharge;  % discharge in kw
array[INTERVALS] of var float: battery_profile;  % discharge in kw


% battery charge or discharge only constraint
% constraint forall (i in INTERVALS) (
%     battery_charge[i] * battery_discharge[i] == 0.0
% );

% battery charge limit in kw
constraint forall (i in INTERVALS) (
    battery_charge[i] >= 0 /\ battery_charge[i] <= power
);

% battery discharge limit in kw
constraint forall (i in INTERVALS) (
    battery_discharge[i] >= 0 /\ battery_discharge[i] <= power
);

% battery profile
constraint forall (i in INTERVALS) (
    battery_profile[i] = battery_charge[i] - battery_discharge[i]
);

% battery capacity constraint in kwh
constraint forall (i in INTERVALS) (
    battery_soc[i] <= max_energy /\ battery_soc[i] >= 0
);

% battery soc constraint
constraint battery_soc[1] = init_energy;
constraint battery_soc[num_intervals] = init_energy;
constraint forall (i in 2..num_intervals) (
    battery_soc[i] = battery_soc[i - 1] + battery_profile[i] / num_intervals_hour
);

% % peak demand constraint   
% constraint forall (i in INTERVALS) (
%     battery_soc[i] *  num_intervals_hour <= max_demand
% );

% objective cost
var float: battery_penalty = sum (i in INTERVALS) (3.37 * 0.00001 * battery_profile[i] * 100 / 0.2);
var float: battery_cost = sum (i in INTERVALS) (
    (loads[i] + battery_profile[i] / num_intervals_hour) * prices[i]
);

solve minimize battery_cost + battery_penalty;
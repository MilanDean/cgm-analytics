import os

import pandas as pd
import matplotlib.pyplot as plt
from simglucose.simulation.user_interface import simulate
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime

## Glucose Simulator - https://github.com/jxx123/simglucose/blob/master/README.md

def run_multiple_scenarios(round):
    # specify start_time as the beginning of today
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    for i in range(10):
        path = './results_{}'.format(round)
        num = '0{}'.format(str(i+1))
        if num == '010':
            num = '10'

        # Create a simulation environment
        patient = T1DPatient.withName('adult#0{}'.format(num))
        sensor = CGMSensor.withName('Dexcom', seed=(i *10*round + 1))
        pump = InsulinPump.withName('Insulet')
        scenario = RandomScenario(start_time=start_time, seed=(i *10*round + 1))
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController()

        # Put them together to create a simulation object
        s1 = SimObj(env, controller, timedelta(days=15), animate=False, path=path)
        results1 = sim(s1)
        # print(results1)
        print('round {} Done!'.format(i+1))

def aggregate_all_subjects(path = './results/'):
    output = []
    files = os.listdir(path)
    for f in sorted(files):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, f))
            df['subject'] = f.strip('.csv')
            output.append(df)

    joined_output = pd.concat(output)
    os.makedirs('../data/synthetic_dataset/')
    joined_output.to_csv('../data/synthetic_dataset/20230605_synthetic_T1DB_dataset.csv', index = False)

def aggregate_all_subjects_3Runs(path = './'):
    output = []
    for round in ['results_1', 'results_2', 'results_3']:
        round_num = round.split('_')[1]
        round_path = os.path.join(path, round)
        files = os.listdir(round_path)
        for f in sorted(files):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(round_path, f))
                subj_num = f.strip('.csv')
                df['subject'] = correct_subj_ID(round_num, subj_num)
                output.append(df)

    joined_output = pd.concat(output)
    os.makedirs('../data/synthetic_dataset/')
    joined_output.to_csv('../data/input/synthetic_dataset/20230611_synthetic_T1DB_dataset.csv', index = False)

def correct_subj_ID(round_num, subj_num):
    round_num = int(round_num)
    if int(round_num) > 1:
        if subj_num == 'adult#010':
            new_subjID = 'adult#0{}'.format(str(10 * round_num))
        else:
            new_subjID = subj_num[:7] + str(round_num-1) + subj_num[8]
    else:
        new_subjID = subj_num

    return new_subjID

if __name__ == '__main__':
    for i in range(1,4):
        round = i
        run_multiple_scenarios(round)
    #     #simulate()
    aggregate_all_subjects_3Runs()

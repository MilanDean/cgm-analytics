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

def run_multiple_scenarios():
    # specify start_time as the beginning of today
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    for i in range(10):
        path = './results'
        num = '0{}'.format(str(i+1))
        if num == '010':
            num = '10'

        # Create a simulation environment
        patient = T1DPatient.withName('adult#0{}'.format(num))
        sensor = CGMSensor.withName('Dexcom', seed=(i *10 + 1))
        pump = InsulinPump.withName('Insulet')
        scenario = RandomScenario(start_time=start_time, seed=(i *10 + 1))
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController()

        # Put them together to create a simulation object
        s1 = SimObj(env, controller, timedelta(days=10), animate=False, path=path)
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

if __name__ == '__main__':
    run_multiple_scenarios()
    simulate()

    ## Testing
    df = pd.read_csv('/Users/dimitriospsaltos/Documents/Personal/Berkeley/w210/cgm-analytics/'
                     'preprocessing/results/2023-06-03_18-24-08/adult#008.csv')
    df.head()
    df['timestamp'] = pd.to_datetime(df.Time)
    plt.scatter(df[df.CHO > 0].timestamp, df[df.CHO > 0].CHO, color='green', label='subject 2 meals')
    plt.plot(df.timestamp, df.CGM, color='green', label='subject 2')
    plt.legend()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import altair_viewer

plt.style.use('ggplot')
# alt.renderers.enable('altair_viewer')

if __name__ == '__main__':

    # Data Load
    file = '../data/input/synthetic_dataset/results/adult#001.csv'
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df.Time)

    # Show timeseries Plot
    plt.figure(figsize = (10,5))
    sns.lineplot(data = df, x = 'timestamp', y='CGM', legend='brief', label = 'CGM')
    plt.title('Raw CGM Data')
    plt.tight_layout()
    plt.show()

    # Processed data
    file = '/Users/dimitriospsaltos/Documents/Personal/Berkeley/w210/cgm-analytics/data/output/training/lgbm_60min/baseline_lgbm_predicitons_20230627.csv'
    preds = pd.read_csv(file)



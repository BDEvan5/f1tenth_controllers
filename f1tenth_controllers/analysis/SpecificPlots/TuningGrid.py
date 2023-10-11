import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_tuning_grid(vehicle_name="TuneMPCC"):
    df = pd.read_csv(f"Logs/{vehicle_name}/PlannerResults_{vehicle_name}.csv")

    df['N'] = df.apply(lambda x: int(x['TestID'].split('x')[0]), axis=1)
    df['SimSteps'] = df.apply(lambda x: int(x['TestID'].split('x')[1]), axis=1)

    print(df)

    data_mean = np.zeros((len(df['N'].unique()), len(df['SimSteps'].unique())))
    data_max = np.zeros((len(df['N'].unique()), len(df['SimSteps'].unique())))
    sim_steps_set = np.sort(df['SimSteps'].unique())
    N_set = np.sort(df['N'].unique())
    print(sim_steps_set)
    print(N_set)
    for i, N in enumerate(N_set):
        for j, sim_steps in enumerate(sim_steps_set):
            progress = df.loc[(df['N'] == N) & (df['SimSteps'] == sim_steps), 'Progress'].values[0]
            if progress > 0.98:
                data_max[i, j] = df.loc[(df['N'] == N) & (df['SimSteps'] == sim_steps), 'TA_max']
                data_mean[i, j] = df.loc[(df['N'] == N) & (df['SimSteps'] == sim_steps), 'TA_mean']
            else:
                data_max[i, j] = np.nan
                data_mean[i, j] = np.nan

    fig = plt.figure(figsize=(6, 3))
    # fig = plt.figure(figsize=(8, 4))
    a1 = plt.subplot(121)
    a2 = plt.subplot(122)
    # a1.matshow(data_mean, cmap='bwr')
    # a1.matshow(data_mean, cmap='YlOrRd')
    # a1.matshow(data_mean, cmap='autumn')
    # a2.matshow(data_max, cmap='summer')
    a1.matshow(data_mean, cmap='RdYlGn')
    a2.matshow(data_max, cmap='RdYlGn')
    # cbar = plt.colorbar()
    # cbar.ax.invert_yaxis()

    a1.xaxis.set_ticks_position('bottom')
    a2.xaxis.set_ticks_position('bottom')
    for a in (a1, a2):
        a.set_xticks(np.arange(len(sim_steps_set)), sim_steps_set*10)
        a.set_yticks(np.arange(len(N_set)), N_set)
        a.set_xlabel("Simulation Period (ms)")
    a1.set_ylabel("N (Prediction Horizon)")

    a1.set_title(f"Mean Tracking Error")
    a2.set_title(f"Max Tracking Error")

    for (i, j), z in np.ndenumerate(data_mean):
        a1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    for (i, j), z in np.ndenumerate(data_max):
        a2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')


    plt.tight_layout()
    # plt.show()
    plt.savefig(f"Logs/{vehicle_name}/TuningGrid_{vehicle_name}.svg", pad_inches=0.05, bbox_inches='tight')
    plt.savefig(f"Logs/{vehicle_name}/TuningGrid_{vehicle_name}.pdf", pad_inches=0.05, bbox_inches='tight')




make_tuning_grid()


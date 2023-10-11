import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_tuning_grid(vehicle_name="TunePointsMPCC3"):
    df = pd.read_csv(f"Logs/{vehicle_name}/PlannerResults_{vehicle_name}.csv")
    df.rename(columns={"TestID": "N"}, inplace=True)
    df = df.sort_values(by=["N"])

    print(df)

    fig = plt.figure(figsize=(5, 2.5))
    
    q1 = df["TA_q1"].values
    q3 = df["TA_q3"].values

    plt.plot(df["N"], df["TA_mean"], 'o-', label="Mean")
    plt.plot(df["N"], df["TA_max"], 'o-', label="Max")
    plt.fill_between(df["N"], q1, q3, alpha=0.2, label="IQR")
    plt.xlabel("N (Prediction Horizon)")
    plt.ylabel("Tracking Error (cm)")
    # plt.ylim([0, 7])
    # plt.xlim([2.5, 16])

    plt.grid()

    x1, x2, y1, y2 = 3.5, 12.5, -0.2, 2.5
    axins = plt.gca().inset_axes([0.35, 0.4, 0.57, 0.52], xlim=(x1, x2), ylim=(y1, y2))

    axins.plot(df["N"], df["TA_mean"], 'o-', label="Mean")
    axins.fill_between(df["N"], q1, q3, alpha=0.2, label="IQR")
    axins.grid(True)

    plt.gca().indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"Logs/{vehicle_name}/TuningPlotN_{vehicle_name}.svg", pad_inches=0.05, bbox_inches='tight')


def make_tuning_grid2(vehicle_name="TunePointsMPCC3"):
    df = pd.read_csv(f"Logs/{vehicle_name}/PlannerResults_{vehicle_name}.csv")
    df.rename(columns={"TestID": "N"}, inplace=True)
    df = df.sort_values(by=["N"])

    print(df)

    fig = plt.figure(figsize=(5, 2.3))
    
    q1 = df["TA_q1"].values
    q3 = df["TA_q3"].values

    plt.plot(df["N"], df["TA_mean"], 'o-', label="Mean")
    plt.plot(df["N"], df["TA_max"], 'o-', label="Max")
    plt.fill_between(df["N"], q1, q3, alpha=0.2, label="IQR")
    plt.xlabel("N (Prediction Horizon)")
    plt.ylabel("Tracking Error (cm)")
    plt.ylim([-0.2, 6.2])
    plt.xlim([2.5, 15.5])

    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.grid()

    # x1, x2, y1, y2 = 3.5, 12.5, -0.2, 2.5
    # axins = plt.gca().inset_axes([0.35, 0.4, 0.57, 0.47], xlim=(x1, x2), ylim=(y1, y2))

    # axins.plot(df["N"], df["TA_mean"], 'o-', label="Mean")
    # axins.fill_between(df["N"], q1, q3, alpha=0.2, label="IQR")
    # axins.grid(True)

    # plt.gca().indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"Logs/{vehicle_name}/TuningPlotN2_{vehicle_name}.svg", pad_inches=0.05, bbox_inches='tight')




# make_tuning_grid2()
make_tuning_grid()


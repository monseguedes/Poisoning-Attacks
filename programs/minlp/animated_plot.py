import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dataframes = []

for i in range(22):
    df = pd.read_csv(f"programs/minlp/attacks/5num5cat/poison_dataframe{i}.csv")
    dataframes.append(df)


fig, ax = plt.subplots()

def update(frame):
    df = dataframes[frame]
    ax.clear()
    for i in range(len(df)):
        ax.scatter(list(df.columns), list(df.iloc[i]))
    ax.set_ylim(-0.1, 1.1) # set the y-axis limit to a fixed range
    ax.set_title(f"Dataframe {frame}")

anim = FuncAnimation(fig, update, frames=len(dataframes), interval=200)
plt.show()

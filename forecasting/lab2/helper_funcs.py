
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [14, 4]
def run_sequence_plot(x, y, title, xlabel="time", ylabel="series", color="k"):
    plt.plot(x, y, color+'-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)


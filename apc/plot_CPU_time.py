from pyomo.common.dependencies.matplotlib import pyplot as plt
# from pyomo.common.dependencies.matplotlib.ticker import MaxNLocator

def _plot_CPU_time(
    online_CPU, offline_CPU, 
    show=False, save=False, fname=None, transparent=False,
):
    fig, ax = plt.subplots()
    ax.set_title("CPU time")
    ax.plot(list(range(len(offline_CPU))), offline_CPU, label="Offline")
    ax.plot(list(range(len(online_CPU))), online_CPU, label="Online")
    ax.legend()
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    if show:
        plt.show()
        
    if save:
        if fname is None:
            fname = "CPU_time.png"
        fig.savefig(fname, transparent=transparent)
        
    return fig, ax
import matplotlib.pyplot as plt
import numpy as np
def create_data(setups: list[dict], seed=None) -> dict:
    rng = np.random.default_rng(seed=seed)
    return {setup["id"]: (setup["b"] - setup["a"]) * rng.random(setup["n"]) + setup["a"] for setup in setups}

def plot_classes(data: dict, save_path: str = None):
    bar_colors = ['black', 'gray', 'red', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'pink', 'purple', 'brown']
    if(data.__len__() != 0):
        _ , ax = plt.subplots(figsize=(8, 8))
        for i, elements in enumerate(data.keys()):
            ax.scatter(data[elements][:,0], data[elements][:,1], c=np.random.choice(bar_colors), label=elements)
        ax.legend()
        ax.set_title(f"2D data showing {data.keys().__len__()} classes")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
    else:
        print("The dict is empty. Exit!")


plot_classes(create_data([
{"id": "classA", "n": (10, 2), "a": 0, "b": 1.5},
{"id": "classB", "n": (20, 2), "a": 3, "b": 4},
{"id": "classC", "n": (25, 2), "a": 0, "b": 10},
{"id": "classD", "n": (20, 2), "a": 2, "b": 7},
], 0))

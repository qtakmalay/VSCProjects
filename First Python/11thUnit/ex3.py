import matplotlib.pyplot as plt
import numpy as np
import dill as pkl

def plot_eval_metrics(data: dict, save_path: str = None):
    num_metrics = len(data)
    _ , ax = plt.subplots(1, num_metrics, figsize=(12, 6))
    cmap = plt.get_cmap('Set2')
    if(num_metrics != 0):
        for i, (metric, inner_dict) in enumerate(data.items()):
            values = inner_dict['values']
            labels = inner_dict['labels']
            num_values = len(values)
            
            bp = ax[i].boxplot(values, labels=labels, patch_artist=True, medianprops={'color': 'black'})
            for j, patch in enumerate(bp['boxes']):
                color = cmap(j%num_values)
                patch.set_facecolor(color)
            ax[i].yaxis.grid(True, linestyle='--', color='gray')
            ax[i].set_yticks(np.arange(0, 1.1, 0.1))
            ax[i].set_ylim(bottom=-0.1, top=1.1)
            ax[i].set_title(metric)
            
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
    else:
        print("empty list exit")

with open("C:\\Users\\azatv\\VSCProjects\\11thUnit\\ex3_data.pkl", "rb") as f:
    ex3_data = pkl.load(f)
plot_eval_metrics(ex3_data, "C:\\Users\\azatv\\VSCProjects\\10thUnit")
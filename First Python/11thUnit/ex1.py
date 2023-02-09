import matplotlib.pyplot as ml
def plot_lifts(data: dict, save_path: str = None):
    if(data.__len__() != 0):
        lifst_bar, ax = ml.subplots(3, 1, figsize=(8, 8))
        ml.ylabel("kg")
        lifst_bar.tight_layout()
        for i, element in enumerate(data.keys()):
            ax[i].set_xticks(range(1, len(data[element])+1))
            ax[i].bar(range(1, len(data[element])+1), data[element])
            ax[i].yaxis.grid()
            ax[i].set_title(element)
            
        ml.show()
        if save_path is not None:
            ml.savefig(save_path)
    else:
        print("The dict is empty. Exit!")
        

lifts = dict()
with open("C:\\Users\\azatv\\VSCProjects\\11thUnit\\ex1_data.csv") as f:
    for line in f.readlines():
        lift_name, weights = line.split(",", maxsplit=1)
        lifts[lift_name] = [int(w) for w in weights.split(",")]
plot_lifts(lifts)
# for element in lifts:
#     print(element)


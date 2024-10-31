import numpy as np
import pickle
from tqdm import tqdm
import sys
import os

f = sys.argv[1]
if os.path.isdir(f):
    max_time = 0
    for file in tqdm(os.listdir(f)):
        with open(os.path.join(f, file), "rb") as infile:
            try:
                data = pickle.load(infile)
            except:
                continue

        hyper = list(data["experiment_data"].keys())[0]
        run_time = data["experiment_data"][hyper]["runs"][0]["train_time"]
        if run_time > max_time:
            max_time = run_time

    print("Maximum run time:", max_time / 3600)

else:
    with open(f, "rb") as infile:
        data = pickle.load(infile)

    time = []
    for hyper in data["experiment_data"]:
        for run in data["experiment_data"][hyper]["runs"]:
            total = run["train_time"] + run["eval_time"]
            time.append(total)

    print("Maximum run time:", np.max(time) / 3600)

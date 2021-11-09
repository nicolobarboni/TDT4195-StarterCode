import torch
import torch.nn as nn
import pathlib
import matplotlib.pyplot as plt
from assignment1.utils import read_im, save_im
import numpy as np
import torchvision
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


model = torch.load("saved_model2.torch")
weight = list(model.children())[1].weight.cpu().data
weight0 = weight[0,:]

weights = []

for i in range(10):
    weights.append(torch.reshape(weight[i, :], (28, 28)))
    save_im(output_dir.joinpath("weights_" + str(i) + ".jpg"), weights[i])


weightplot0 = plt.imshow(weights[0])
plt.show()
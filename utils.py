import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from matplotlib import pyplot
from typing import List

def load_1st_conv_layers(model):

    model_layers = list(model.children())

    for i in range(len(model_layers)):
        if str(model_layers[i]).startswith("Conv2d"):

            return model_layers[i]

def generate_feature_maps(model, file: str):

    layer_name = load_1st_conv_layers(model)

    img = cv2.imread('static/imgs/uploads/{}'.format(file))
    if img.shape[0] != 224 or img.shape[1] != 224:
        img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.swapaxes(img_tensor, 1, 2)
    img_tensor = torch.swapaxes(img_tensor, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).float()

    # generate the feature map
    feature_map = layer_name(img_tensor).cpu().detach()
    feature_map = np.array(feature_map, dtype=float)

    pyplot.figure(figsize=(20, 10))
    fig, axs = plt.subplots(3, 4)
    fig.patch.set_facecolor('#0a011c')


    for position in list(np.arange(1, 13)):
        pyplot.subplot(3, 4, position)

        pyplot.imshow(feature_map[0][position -1], cmap='viridis')
        pyplot.imshow(feature_map[0][position -1], cmap='viridis')
        pyplot.axis("off")

    # decrease space between subplots
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=-0.2)
    fig.suptitle('Feature maps', size = 18, color ='white', y = 0.94)
    fig.savefig('static/imgs/feature_maps/{}.png'.format(file.split(".")[0]))

def get_probability(model, file: str) -> List[float]:

    softmax = nn.Softmax()

    img = cv2.imread('static/imgs/uploads/{}'.format(file))
    if img.shape[0] != 224 or img.shape[1] != 224:
        img = cv2.resize(img, (224, 224))

    img_tensor = torch.from_numpy(img)
    img_tensor = torch.swapaxes(img_tensor, 1, 2)
    img_tensor = torch.swapaxes(img_tensor, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).float()

    prediction = model(img_tensor)
    prediction = torch.round(softmax(prediction), decimals = 3)
    prediction = prediction.squeeze(0).cpu().detach().numpy().tolist()

    return prediction

def plot_model_prediction(prediction: List[float], file: str):

    fig = plt.figure(figsize=(20, 12.5))
    fig.patch.set_facecolor('#0a011c')

    ax = plt.axes()

    # Setting the background color of the plot
    # using set_facecolor() method
    ax.set_facecolor("#0a011c")
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['top'].set_color('#dddddd')
    ax.spines['right'].set_color('#dddddd')
    ax.spines['left'].set_color('#dddddd')

    classes = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower',
     'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    for list_position, name in enumerate(classes):
        if "_" in name:
            name = name.replace("_", " ")
            classes[list_position] = name

    choice = classes[np.argmax(np.array(prediction))]

    plt.title('Model prediction: {}'.format(choice), size = 50, pad=50, color='white')
    plt.xlabel('Category', weight = 'bold', fontsize = 30, color='white')
    plt.ylabel('Probability', weight = 'bold', fontsize = 30, labelpad = 40, color='white')
    plt.xticks(rotation=80, fontsize = 20, color ='white')
    plt.yticks(fontsize = 20, color='white')
    plt.subplots_adjust(bottom=0.2)

    plt.bar(classes, prediction)
    plt.ylim([0, 1])

    # plot probability values in figure
    for i, value in enumerate(prediction):
        print(i, value)
        if value > 0.95:
            plt.text(s=round(value, 2), x=i - 0.28, y=value - 0.03, size=20, color='white')
        else:
            plt.text(s=round(value, 2), x= i - 0.28, y=value + 0.01, size = 20, color='white')

    fig.savefig('static/imgs/model_prediction/model_prediction_{}.png'.format(file.split(".")[0]))
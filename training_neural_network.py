import cv2, os
import torch
import torch.nn as nn
from neural_network import Net
from torch.utils.data import Dataset, DataLoader
import json
import time
import torch.optim as optim
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device is: {}" .format(device))

amount_of_epochs = 5
save_path = 'models/'

class CustomDataset(Dataset):
    def __init__(self, path):
        self.main_route = 'vegatable_data/{}'.format(path)
        self.data = []
        self.labels_dict = {}
        self.folders = os.listdir(self.main_route)
        self.vegatable_folders = os.listdir(self.main_route)
        self.path_to_veggie_folders = [self.main_route + "/" + folder for folder in self.vegatable_folders]

        for i, folder in enumerate(self.folders):
            self.labels_dict[folder] = i

        # loop over the vegatable folders
        for path_to_veggie_folder in self.path_to_veggie_folders:
            # loop over the vegatable images per folder
            for veggie_img in os.listdir(path_to_veggie_folder):
                self.veggie_folder_name = path_to_veggie_folder.split("/")[-1]
                self.path = path_to_veggie_folder + "/" + veggie_img
                self.data.append([self.path, self.labels_dict[self.veggie_folder_name]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path, label = self.data[idx]

        img = cv2.imread(img_path)

        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224))

        if random.uniform(0, 1) < 0.5 and net.training == True:

            # flip horizontally during training
            img = cv2.flip(img, 0)

        img_tensor = torch.from_numpy(img)
        img_tensor = torch.swapaxes(img_tensor, 1, 2)
        img_tensor = torch.swapaxes(img_tensor, 0, 1)

        label = torch.tensor([label])

        return img_tensor.float(), label.float()

def save_results_after_training(training_accuracies, validation_accuracies, training_losses, validation_losses):

    with open(save_path + "training_accuracies_" + time.strftime("%Y%m%d-%H%M%S"), "w") as fp:
        json.dump(training_accuracies, fp)
    with open(save_path + "validation_accuracies" + time.strftime("%Y%m%d-%H%M%S"), "w") as fp:
        json.dump(validation_accuracies, fp)

    with open(save_path + "training_loss" + time.strftime("%Y%m%d-%H%M%S"), "w") as fp:
        json.dump(training_losses, fp)
    with open(save_path + "validation_losses" + time.strftime("%Y%m%d-%H%M%S"), "w") as fp:
        json.dump(validation_losses, fp)

def evaluate(data_loader_validate):

    if not os.path.exists(save_path):
        os.makedir(save_path)

    val_loss = 0.0
    val_accuracy = 0.0

    for validate_i, (imgs, labels) in enumerate(data_loader_validate):

        net.eval()

        with torch.no_grad():
            labels = labels.to(device)
            imgs = imgs.to(device)

            predictions = net(imgs).to(device)
            loss = criterion(predictions, labels.squeeze().long())

            max_predictions = predictions.argmax(1).type(torch.LongTensor).to(device)
            correct_label = labels.squeeze().type(torch.LongTensor).to(device)

            accuracy = float(torch.sum(max_predictions == correct_label)) / predictions.shape[0]
            val_accuracy += accuracy
            val_loss += loss.item()

    net.train()

    return val_accuracy / validate_i, val_loss / validate_i

def import_data_loaders():
    train_data = CustomDataset(path = 'train/')
    data_loader_train = DataLoader(dataset = train_data, batch_size = 32, shuffle = True, drop_last = True)

    validate_data = CustomDataset(path = 'validation/')
    data_loader_validate = DataLoader(dataset=validate_data, batch_size=32, shuffle=True, drop_last=True)

    return data_loader_train, data_loader_validate


if __name__ == "__main__":

    net = Net().to(device)

    data_loader_train, data_loader_validate = import_data_loaders()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum = 0.9)

    validation_acc, validation_loss = evaluate(data_loader_validate)
    print("validation accuracy before training: {}, validation loss before training: {}".format(
                                                                round(validation_acc, 2),
                                                                round(validation_loss, 2)))

    training_accuracies = []
    validation_accuracies = []
    training_losses = []
    validation_losses = []

    for epoch in range(amount_of_epochs):

        print("epoch: {}".format(epoch))

        training_loss = 0.0
        training_acc = 0.0

        for train_i, (imgs, labels) in enumerate(data_loader_train):

            labels = labels.to(device)
            imgs = imgs.to(device)

            optimizer.zero_grad()

            predictions = net(imgs).to(device)
            loss = criterion(predictions, labels.squeeze().long())
            loss.backward()

            optimizer.step()

            training_loss += loss.item()

            max_predictions = predictions.argmax(1).type(torch.LongTensor).to(device)
            correct_label = labels.squeeze().type(torch.LongTensor).to(device)

            accuracy = float(torch.sum(max_predictions == correct_label)) / predictions.shape[0]
            training_acc += accuracy

            if train_i % 250 == 1:
                print("training step: {}".format(train_i))

                print("training accuracy: {}, training loss: {}".format(round((training_acc / train_i), 2),
                                                                        round(training_loss / train_i, 2)))


        validation_acc, validation_loss = evaluate(data_loader_validate)

        print("validation accuracy: {}, validation loss: {}".format(round(validation_acc, 2),
                                                                round(validation_loss, 2)))

        training_accuracies.append(training_acc / train_i)
        validation_accuracies.append(validation_acc)
        training_losses.append(training_loss / train_i)
        validation_losses.append(validation_loss)

    save_results_after_training(training_accuracies, validation_accuracies, training_losses, validation_losses)

    torch.save(net.state_dict(), 'models/trained_model_epoch{}_time_{}.pth'.format(amount_of_epochs, time.strftime("%Y%m%d-%H%M%S")))

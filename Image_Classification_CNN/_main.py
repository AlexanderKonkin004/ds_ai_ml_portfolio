from dataset import ImagesDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset
import numpy as np
import torch
from architecture import MyCNN
from architecture import model
from training_loop import training_loop
from augmentation import TransformedImagesDataset

# 0 # - Set seed
seed = 101
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1 # - Create a dataset
img_dir = "training_data"
dataset = ImagesDataset(img_dir)
dataset = TransformedImagesDataset(dataset)

# 2 # - Split the data
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 3 # - Train the model

loss = torch.nn.functional.cross_entropy

losses = training_loop(
    network=model,
    train_data=train_dataset,
    eval_data=val_dataset,
    loss=loss,
    num_epochs=15,
    batch_size=128,
    learning_rate=0.001,
    #momentum=0.9
)

# 4 # - Test the model

#saved_model = MyCNN()
#saved_model_path = "model.pth"
#saved_model.load_state_dict(torch.load(saved_model_path))
#
#from torch.utils.data import DataLoader
#test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
#
#print(eval.evaluate_model(saved_model, test_loader, loss=loss, accuracy=eval.multiclass_accuracy))

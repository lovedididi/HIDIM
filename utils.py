import random

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    """
    set fixed seed
    :param seed:
    :return:
    """
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Get the number of classes and the total number of samples in the dataset
def get_c_num(Dataset):
    total_c_num = Dataset.__len__()
    c_num_dict = {}
    list_classes = Dataset.targets
    for c, num in Dataset.class_to_idx.items():
        c_num_dict[num] = list_classes.count(num)
    return total_c_num, c_num_dict

# Get the expert output corresponding to the respective category
def get_tc_ic_outputs(output, expert_dict_class, i, Cs_index, labels):
    list_tc_c = expert_dict_class[i]
    list_index_tc = []
    list_index_ic = []

    for c, c_index in Cs_index.items():
        if c in list_tc_c:
            list_index_tc += c_index
        else:
            list_index_ic += c_index
    if len(list_index_tc) != 0:
        tc_tensor_outputs = torch.empty(len(list_index_tc), output.shape[1])
        tc_tensor_labels = torch.empty(len(list_index_tc), output.shape[1], dtype=torch.float64)
        index_count = 0
        for index in list_index_tc:
            tc_tensor_outputs[index_count] = output[index]
            tc_tensor_labels[index_count] = labels[index]
            index_count += 1
    else:
        tc_tensor_outputs = None
        tc_tensor_labels = None

    if len(list_index_ic) != 0:
        ic_tensor_outputs = torch.empty(len(list_index_ic), output.shape[1])
        index_count_ic = 0
        for index in list_index_ic:
            ic_tensor_outputs[index_count_ic] = output[index]
            index_count_ic += 1
    else:
        ic_tensor_outputs = None
    return tc_tensor_outputs, tc_tensor_labels, ic_tensor_outputs

def get_c_one_index(tensor):
    numpy_array = tensor.cpu().numpy()
    experts_index = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    for i, num in enumerate(numpy_array):
        if experts_index[num] == None:
            experts_index[num] = [i]
        else:
            experts_index[num].append(i)

    new_experts_index = {}
    for key, value in experts_index.items():
        if value != None:
            new_experts_index[key] = experts_index[key]
    return new_experts_index



# Get the index
def get_c_index(tensor):
    _, label_one = torch.max(tensor, dim=1)
    numpy_array = label_one.cpu().numpy()
    experts_index = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    for i, num in enumerate(numpy_array):
        if experts_index[num] == None:
            experts_index[num] = [i]
        else:
            experts_index[num].append(i)

    new_experts_index = {}
    for key, value in experts_index.items():
        if value != None:
            new_experts_index[key] = experts_index[key]
    return new_experts_index



class LWSClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(LWSClassifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.ones(1, out_features))

    def forward(self, x):
        logits = self.fc(x)
        w = self.fc.weight.data
        scaled_logits = logits * self.scale
        return scaled_logits



def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class MyData(ImageFolder):
    cls_num = 7  # 类别数量

    def __init__(self,
                 root: str,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None):
        super().__init__(root, transform=transform,
                         target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        self.gen_imbalanced_data()

    def gen_imbalanced_data(self) -> None:
        targets_np = np.array(self.targets, dtype=np.int64)
        self.one_hot_labels = to_categorical(targets_np, num_classes=self.cls_num)

    def __getitem__(self, index: int):
        sample, _ = super().__getitem__(index)
        return sample, self.one_hot_labels[index]


class FocalLoss(nn.Module):
    """
    FL(pt) = -α * (1 - pt)^γ * log(pt)
    """

    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (B, C)
            targets: (B,) or (B, C)
        """
        if targets.dim() == 1:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = -torch.sum(targets * F.log_softmax(inputs, dim=1), dim=1)


        pt = torch.exp(-ce_loss)


        focal_loss = self.alpha * (1 - pt + self.eps) ** self.gamma * ce_loss
        return focal_loss.mean()
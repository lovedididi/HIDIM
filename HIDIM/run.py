import os

import torch

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import resnet50
from new_expert import expert_resnet50

from utils import get_tc_ic_outputs, get_c_one_index, get_c_index, MyData, set_seed, FocalLoss
from Classification import New_Expert_Train, New_Expert_validate

def train(train_loader, model, optimizer, epoch, fl):
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    model.train()
    print(f"train epoch: {epoch}/{epochs}")
    for step, (images, labels) in enumerate(train_loader, 0):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, False, isExpert4=False)
        optimizer.zero_grad()

        output = sum(outputs) / len(outputs)
        loss = fl(output, labels)

        _, predicted = torch.max(output.data, 1)
        _, target = torch.max(labels.data, 1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += torch.sum(predicted == target.data)
        train_total += labels.size(0)
    return train_loss, train_correct, train_total

def validate(test_loader, model, cost):
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = sum(outputs) / len(outputs)

            loss = cost(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += torch.sum(predicted == labels.data)
            test_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    return test_loss, test_correct, f1, test_total




if __name__ == '__main__':
    transform = transforms.Compose([ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]
                                    ),
                                    transforms.Resize((224, 224))
                                    ])
    set_seed(129)
    data_dir_train = '/data/wheat_r1-r14_M600/train'
    data_dir_test = '/data/wheat_r1-r14_M600/test'

    #data_dir_train = "/data/TRAIN-Wheat-P600-1/train"
    #data_dir_test = "/data/TRAIN-Wheat-P600-1/WHEAT_R1-14_P600_TEST"

    #data_dir_train = '/data/wheat_r1-r14_g600/WHEAT_R1-14_G600_TRAIN1OF3'
    #data_dir_test = '/data/wheat_r1-r14_g600/WHEAT_R1-14_G600_TEST'

    parent_dir = os.path.dirname(data_dir_train)
    prefix = os.path.basename(parent_dir)

    TrainDataset = MyData(root=data_dir_train, transform=transform)
    TestDataset = datasets.ImageFolder(data_dir_test, transform=transform)
    Train_Len = len(TrainDataset)
    Test_Len = len(TestDataset)

    epochs = 150
    batch_size = 128
    # class number
    label_dis = 7
    # Categories under the responsibility of M+1 experts
    expert2_tl = [0, 1, 2, 3, 4, 6]
    expert_dict_class = {1: expert2_tl}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=TrainDataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=TestDataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = resnet50(num_classes=label_dis, num_exps=3, use_norm=True).to(device)
    model_Classification = expert_resnet50(num_classes=label_dis, num_exps=1, use_norm=False).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    cost = torch.nn.CrossEntropyLoss()
    fl = FocalLoss(alpha=0.25, gamma=3.0)
    optimizer_Classification = torch.optim.Adam(model_Classification.parameters())

    best_f1 = 0.0
    for epoch in range(epochs):
        train_loss, train_correct, train_total = train(train_loader, model, optimizer, epoch, fl)

        test_loss, test_correct, f1, test_total = validate(test_loader, model, cost)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f'best_f1_{prefix}.pth')

        print(
            "Train Loss is:{:.8f}, Train Accuracy is:{:.4f}%, Test Loss is::{:.8f} Test Accuracy is:{:.4f}% F1 Score is:{:.4f}".format(
                train_loss / Train_Len, 100 * train_correct / Train_Len,
                test_loss / Test_Len, 100 * test_correct / Test_Len, f1))
    print("The first phase has been completed")

    model.load_state_dict(torch.load(f'best_f1_{prefix}.pth'))
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.eval()
    for epoch in range(epochs):
        print(f"train epoch: {epoch}/{epochs}")
        New_Expert_Train(model, model_Classification, train_loader, fl, device, expert_dict_class, optimizer_Classification, Train_Len)
        New_Expert_validate(model, model_Classification, test_loader, cost, device, Test_Len)
    print("Completed")

import torch

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import resnet50
from new_expert import expert_resnet50

from utils import get_tc_ic_outputs, get_c_one_index, get_c_index, MyData, set_seed, FocalLoss

def New_Expert_Train(model, model_expert4, train_loader, fl, device, expert_dict_class, optimizer, Train_Len):
    expert4_loss = 0.0
    correct = 0.0

    model_expert4.train()
    for step, (images, labels) in enumerate(train_loader, 0):
        images, labels = images.to(device), labels.to(device)
        output = model_expert4(images, model)
        optimizer.zero_grad()

        Cs_index = get_c_index(labels)
        output_tc, label_tc, output_ic = get_tc_ic_outputs(output, expert_dict_class, 1, Cs_index, labels)

        loss_output_tc = fl(output_tc, label_tc) if output_tc is not None else 0.0
        loss_output_ic = torch.sum(output_ic ** 2) if output_ic is not None else 0.0
        loss = loss_output_tc + 0.001 * loss_output_ic
        loss.backward()

        optimizer.step()
        expert4_loss += loss.item()
        if output_tc != None:
            _, target = torch.max(label_tc.data, 1)
            _, predicted = torch.max(output_tc.data, dim=1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
    print("Train Loss is:{:.8f}, Train Accuracy is:{:.4f}%".format(expert4_loss / Train_Len, 100 * correct / Train_Len))


def New_Expert_validate(model, model_Classification, test_loader, cost, device, Test_Len):
    model_Classification.eval()
    correct = 0
    Classification_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            Cs_index = get_c_one_index(labels)
            MoE_classes = []
            Classification_classes = []
            # Judging that it is not a normal class
            for c, indexs in Cs_index.items():
                if c != 5:
                    Classification_classes += indexs
                else:
                    MoE_classes += indexs

            outputs = model(images, False, False)
            output = sum(outputs) / len(outputs)
            if MoE_classes != []:
                MoE_normal_outputs = torch.empty(len(MoE_classes), output.shape[1])
                Normal_Labels = torch.empty(len(MoE_classes), dtype=torch.int64)

                num_index = 0
                for index in MoE_classes:
                    MoE_normal_outputs[num_index] = output[index]
                    Normal_Labels[num_index] = labels[index]
                    num_index += 1
                MoE_loss = cost(MoE_normal_outputs, Normal_Labels)
                Classification_loss += MoE_loss.item()
                _, predicted = torch.max(MoE_normal_outputs.data, dim=1)
                correct += predicted.eq(Normal_Labels.view_as(predicted)).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(Normal_Labels.cpu().numpy())
            if Classification_classes != []:
                Classification_outputs = model_Classification(images, model).to(device)

                MoE_abnormal_outputs = torch.empty(len(Classification_classes), output.shape[1])
                Abnormal_Labels = torch.empty(len(Classification_classes), dtype=torch.int64)
                Classification_Abnormal_outputs = torch.empty(len(Classification_classes), output.shape[1])
                num_index = 0
                for index in Classification_classes:
                    Classification_Abnormal_outputs[num_index] = Classification_outputs[index]
                    MoE_abnormal_outputs[num_index] = output[index]
                    Abnormal_Labels[num_index] = labels[index]
                    num_index += 1
                avg_outputs = 0.6 * Classification_Abnormal_outputs + 0.4 * MoE_abnormal_outputs
                loss_Abnormal = cost(avg_outputs, Abnormal_Labels)
                Classification_loss += loss_Abnormal.item()
                _, predicted = torch.max(avg_outputs.data, dim=1)
                correct += predicted.eq(Abnormal_Labels.view_as(predicted)).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(Abnormal_Labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    print("Test Loss is::{:.8f} Test Accuracy is:{:.4f}% F1 Score is:{:.4f}".format(Classification_loss / Test_Len, 100 * correct / Test_Len, f1))



if __name__ == '__main__':
    transform = transforms.Compose([ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]
                                    ),
                                    transforms.Resize((224, 224))
                                    ])
    set_seed(129)
    #data_dir_train = '/data/wheat_r1-r14_M600/train'
    #data_dir_test = '/data/wheat_r1-r14_M600/test'


    data_dir_train = "/data/TRAIN-Wheat-P600-1/train"
    data_dir_test = "/data/TRAIN-Wheat-P600-1/WHEAT_R1-14_P600_TEST"

    #data_dir_train = '/data/wheat_r1-r14_g600/WHEAT_R1-14_G600_TRAIN1OF3'
    #data_dir_test = '/data/wheat_r1-r14_g600/WHEAT_R1-14_G600_TEST'
    TrainDataset = MyData(root=data_dir_train, transform=transform)
    TestDataset = datasets.ImageFolder(data_dir_test, transform=transform)

    Train_Len = len(TrainDataset)
    Test_Len = len(TestDataset)

    batch_size = 128
    epochs = 150
    label_dis = 7
    expert2_tl = [0, 1, 2, 3, 4, 6]
    expert_dict_class = {1: expert2_tl}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=TrainDataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=TestDataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = resnet50(num_classes=label_dis, num_exps=3, use_norm=True).to(device)
    model_Classification = expert_resnet50(num_classes=label_dis, num_exps=1, use_norm=False).to(device)

    cost = torch.nn.CrossEntropyLoss()
    fl = FocalLoss(alpha=0.25, gamma=3.0)

    optimizer_Classification = torch.optim.Adam(model_Classification.parameters())
    model.load_state_dict(torch.load('best_f1_model_p600.pth', map_location=device))
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.eval()
    for epoch in range(epochs):
        print(f"train epoch: {epoch}/{epochs}")
        New_Expert_Train(model, model_Classification, train_loader, fl, device, expert_dict_class, optimizer_Classification, Train_Len)
        New_Expert_validate(model, model_Classification, test_loader, cost, device, Test_Len)
    print("Completed")





import torch
import torch.optim as optim
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from PIL import Image
from config import AppConfig
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

class BeansDataset(torch.utils.data.Dataset):
    def __init__(self, stage, root_dir, csv_samples_info=None, transform=None):

        df = pd.read_csv(csv_samples_info)
        if stage=='train':
            self.landmarks_frame = df[df['data set'] == 'train']
        elif stage=='val':
            self.landmarks_frame = df[df['data set'] == 'test']
        else:
            self.landmarks_frame = df

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_pth = str(self.root_dir / self.landmarks_frame.iloc[idx]['filepaths'])
        image = Image.open(img_pth).convert('RGB')
        class_id =  self.landmarks_frame.iloc[idx]['class index']
        class_name = self.landmarks_frame.iloc[idx]['labels']

        if self.transform:
            image = self.transform(image)

        return image, class_id


def train(config, model, device, train_loader, optimizer, epoch, writer):

    running_loss = 0
    running_corrects = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        running_corrects += torch.sum(torch.max(output,1)[1] == target)

    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #         epoch, batch_idx * len(data), len(train_loader.dataset),
    #         100. * batch_idx / len(train_loader), loss.item()))
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print(f'Epoch: {epoch}')
    print(f'Train. Loss: {epoch_loss:.4f}. Acc: {epoch_acc:.4f}')
    writer.add_scalar('training loss',
                      epoch_loss,
                      epoch * len(train_loader))
    writer.add_scalar('training acc',
                      epoch_acc,
                      epoch * len(train_loader))

def validate(model, device, val_loader, epoch, writer):

    running_loss = 0
    running_corrects = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target) # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(torch.max(output, 1)[1] == target)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    print(f'Val. Loss: {epoch_loss:.4f}. Acc: {epoch_acc:.4f}')
    writer.add_scalar('val loss',
                      epoch_loss,
                      epoch * len(val_loader))
    writer.add_scalar('val acc',
                      epoch_acc,
                      epoch * len(val_loader))


def test(model, config: AppConfig):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    writer = SummaryWriter("runs/CoffeBeans")
    test_kwconfig = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwconfig = {'num_workers': 1,
                         'pin_memory': True,
                         'shuffle': True}
        test_kwconfig.update(cuda_kwconfig)

    running_loss = 0
    running_corrects = 0

    model.eval()

    dataset0 = BeansDataset(stage='all',
                            root_dir=config.dataset_output_path,
                            csv_samples_info=str(config.dataset_output_path /
                                                 "Coffee Bean.csv"),
                            transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset0, **test_kwconfig)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(torch.max(output, 1)[1] == target)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test. Loss: {epoch_loss:.4f}. Acc: {epoch_acc:.4f}')
    writer.add_scalar('test loss',
                      epoch_loss,
                      len(test_loader))
    writer.add_scalar('test acc',
                      epoch_acc,
                      len(test_loader))

    return epoch_loss, epoch_acc

def main_actions(config: AppConfig):

    writer = SummaryWriter("runs/CoffeBeans")

    use_cuda = not config.no_cuda and torch.cuda.is_available()

    torch.manual_seed(config.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    dataset0 = BeansDataset(stage='all',
                            root_dir=config.dataset_output_path,
                            csv_samples_info=str(config.dataset_output_path /
                                                 "Coffee Bean.csv"),
                            transform=transform)

    train_kwconfig = {'batch_size': config.batch_size}
    val_kwconfig = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwconfig = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwconfig.update(cuda_kwconfig)
        val_kwconfig.update(cuda_kwconfig)


    dataset1 = BeansDataset(stage='train',
                            root_dir=config.dataset_output_path,
                            csv_samples_info=str(config.dataset_output_path /
                                                 "Coffee Bean.csv"),
                            transform=transform)
    dataset2 = BeansDataset(stage='val',
                            root_dir=config.dataset_output_path,
                            csv_samples_info=str(config.dataset_output_path /
                                                 "Coffee Bean.csv"),
                            transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwconfig)
    val_loader = torch.utils.data.DataLoader(dataset2, **val_kwconfig)

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=4, bias=True)
    model.to(device)
    #model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, epoch, writer)
        validate(model, device, val_loader, epoch, writer)
        scheduler.step()

    return model
    # if config.save_model:
    #     torch.save(model.state_dict(), "beans_resnet18.pt")


if __name__ == '__main__':
    app_config = AppConfig.parse_raw()
    print(app_config)
    model_ft = main_actions(app_config)
    test(model_ft, app_config)
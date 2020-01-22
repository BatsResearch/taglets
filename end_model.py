import torch
import numpy as np
import torch.nn.functional as F
import torchvision


class EndModel(torch.nn.Module):
    """
    EndModel class
    """
    def __init__(self):
        super(EndModel, self).__init__()
        pass

    def optimize(self, train_dataloader, test_dataloader, use_gpu=True):
        """
        Optimize model
        """
        pass


class MNISTNet(EndModel):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(9216, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 10)
        )
        self.trained = False

    def forward(self, x):
        x = torch.flatten(self.conv(x), 1)
        x = self.linear(x)
        return x

    def optimize(self, train_dataloader, test_dataloader, use_gpu=True, save_path='./MNISTNet.ptm'):

        def soft_cross_entropy(input, target):
            logs = torch.nn.LogSoftmax(dim=1)
            return torch.mean(torch.sum(-target * logs(input), 1))

        epoch = 20
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        if use_gpu:
            self.cuda()

        for i in range(epoch):
            # Train
            self.train()
            tlosses = []
            tacc = []
            for img, label in train_dataloader:
                optimizer.zero_grad()

                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()

                predictions = self(img)
                loss = soft_cross_entropy(predictions, label)
                loss.backward()
                optimizer.step()

                accuracy = torch.sum(torch.max(predictions, 1)[1] == torch.max(label, 1)[1]).item() / float(len(label))
                tacc.append(accuracy)
                tlosses.append(loss.item())

            # Eval
            self.eval()
            eacc = []
            for img, label in test_dataloader:

                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()

                predictions = self(img)
                accuracy = torch.sum(torch.max(predictions, 1)[1] == torch.max(label, 1)[1]).item() / float(len(label))
                eacc.append(accuracy)

            print("Epoch: {:d} =| Train Loss: {:f}  Acc: {:f} || Test Acc: {:f}".format(i+1,
                                                                                        np.mean(tlosses),
                                                                                        np.mean(tacc),
                                                                                        np.mean(eacc)))
            scheduler.step()
        torch.save(self.state_dict(), save_path)

    def test(self, eval_dataloader, use_gpu=True, save_path='./MNISTNet.ptm'):
        self.load_state_dict(torch.load(save_path))
        pass


def main():
    end_model = MNISTNet()
    mnist = torchvision.datasets.MNIST('~/Dataset/', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())

    def shuffle(mnist, n, ratio=0.9):
        def to_soft_one_hot(l):
            soh = [0.15] * 10
            soh[l] = 0.85
            return torch.FloatTensor(soh)

        # 80 train 20 test
        train_thresh = int(ratio * n)
        idx = [x for x in range(n)]
        dataset = torch.utils.data.Subset(mnist, idx)
        data = []
        for i, l in dataset:
            data.append((i, to_soft_one_hot(l)))

        train_dataset = torch.utils.data.DataLoader(data[:train_thresh], batch_size=256, shuffle=True)
        test_dataset = torch.utils.data.DataLoader(data[train_thresh:], batch_size=128, shuffle=True)

        return train_dataset, test_dataset

    train, test = shuffle(mnist, 20000)
    end_model.optimize(train, test)


if __name__ == '__main__':
    main()

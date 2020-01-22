import torch
import numpy as np
import torch.nn.functional as F

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
        output = F.log_softmax(x, dim=1)
        return output

    def optimize(self, train_dataloader, test_dataloader, use_gpu=True, cfg=None, save_path='./MNISTNet.ptm'):

        def soft_cross_entropy(input, target):
            logs = torch.nn.LogSoftmax(dim=1)
            return torch.mean(torch.sum(-target * logs(input), 1))

        epoch = 30

        optimizer = torch.optim.Adam(self.parameters())

        if use_gpu:
            self.cuda()

        for i in range(epoch):
            # Train
            self.train()
            tlosses = []
            for img, label in train_dataloader:
                optimizer.zero_grad()

                img = torch.FloatTensor(img).cuda()
                label = torch.FloatTensor(label).cuda()

                preds = self(img)
                loss = soft_cross_entropy(preds, label)
                loss.backward()

                optimizer.step()
                tlosses.append(loss.item())

            # Eval
            self.eval()
            elosses = []
            for img, label in test_dataloader:

                img = torch.FloatTensor(img).cuda()
                label = torch.FloatTensor(label).cuda()

                preds = self(img)
                loss = soft_cross_entropy(preds, label)
                elosses.append(loss.item())

            print("Epoch: {:d} =| Train Loss: {:f}   ||    Test Loss: {:f}".format(i+1,
                                                                                   np.mean(tlosses),
                                                                                   np.mean(elosses)))

        torch.save(self.state_dict(), save_path)

    def test(self, eval_dataloader, use_gpu=True, save_path='./MNISTNet.ptm'):
        pass


def main():
    import torchvision

    end_model = MNISTNet()

    mnist = torchvision.datasets.MNIST('~/Dataset/', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())

    def shuffle(mnist, n, ratio=0.8):
        def tosoft_onehot(l):
            soh = [0.15] * 10
            soh[l] = 0.85
            return torch.FloatTensor(soh)


        # 80 train 20 test
        train_thresh = int(ratio*n)
        #idx = [np.random.randint(len(mnist)) for _ in range(n)]
        idx = [x for x in range(n)]
        dataset = torch.utils.data.Subset(mnist, idx)
        data = []
        for i, l in dataset:
            data.append((i, tosoft_onehot(l)))



        train_dataset = torch.utils.data.DataLoader(data[:train_thresh], batch_size=128, shuffle=True)
        test_dataset = torch.utils.data.DataLoader(data[train_thresh:], batch_size=128, shuffle=True)

        return train_dataset, test_dataset

    traindl, testdl = shuffle(mnist, 20000)

    end_model.optimize(traindl, testdl)








if __name__ == '__main__':
    main()






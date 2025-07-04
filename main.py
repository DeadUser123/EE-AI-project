if __name__ == "__main__":
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    # https://www.youtube.com/watch?v=vBlO87ZAiiw
    train_data = datasets.MNIST(
        root="data",
        train = True,
        transform=ToTensor(),
        download = True,
    )

    test_data = datasets.MNIST(
        root="data",
        train = False,
        transform=ToTensor(),
        download = True,
    )

    print(train_data)

    from torch.utils.data import DataLoader
    loaders = {
        "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
    }

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    class CNN(nn.Module):
        def __init__(self, *args, **kwds):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, 5)
            self.conv2 = nn.Conv2d(10, 20, 5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            
            return F.softmax(x)
        
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(loaders["train"]):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx / len(loaders['train']):.0f}%)]\tLoss: {loss.item():.6f}")
        
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loaders["test"]:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(loaders["test"].dataset)
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.0f}%)\n")

    for epoch in range(1, 11):
        train(epoch)
        test()
        
    from torch.utils.data import TensorDataset

    # Step 1: Generate pseudo-labels from the model
    def generate_pseudo_labels(data_loader):
        model.eval()
        pseudo_images = []
        pseudo_labels = []

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                pseudo_images.append(data.cpu())
                pseudo_labels.append(pred.cpu()) # use predictions as the labels

        pseudo_images = torch.cat(pseudo_images)
        pseudo_labels = torch.cat(pseudo_labels)
        return pseudo_images, pseudo_labels

    # Generate pseudo-labeled dataset
    pseudo_images, pseudo_labels = generate_pseudo_labels(loaders["train"])

    pseudo_dataset = TensorDataset(pseudo_images, pseudo_labels)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=100, shuffle=True)

    # Step 2: Train on pseudo-labeled data
    def self_train(epoch_start, epoch_end):
        for epoch in range(epoch_start, epoch_end + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(pseudo_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 20 == 0:
                    print(f"[Self-Training] Epoch: {epoch} [{batch_idx * len(data)}/{len(pseudo_loader.dataset)} "
                        f"({100. * batch_idx / len(pseudo_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            test()

    # Self-train for 11 more epochs
    self_train(12, 22)
        
    import matplotlib.pyplot as plt

    model.eval()

    data, target = test_data[0]
    data = data.unsqueeze(0).to(device)

    output = model(data)
    prediction = output.argmax(dim=1, keepdim=True).item()

    print(f"Predicted label: {prediction}, True label: {target}")
    image = data.squeeze(0).squeeze(0).cpu().numpy()

    plt.imshow(image, cmap='gray')
    plt.show()
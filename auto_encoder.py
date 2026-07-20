import argparse
import datetime
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

from torchvision import datasets, transforms


class Encoder(nn.Module):
    def __init__(self, latent_dims=128):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 4, 3, 1, padding=1)       # 28 x 28
        self.conv1 = nn.Conv2d(4, 8, 3, 2, padding=1)       # 14 x 14
        self.conv2 = nn.Conv2d(8, 16, 3, 2, padding=1)      # 7 x 7
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(784, latent_dims)

    def forward(self, x):
        #print(self.__class__.__name__, 'input', x.shape)
        x = self.conv0(x)
        #print(self.__class__.__name__, 'conv0', x.shape)
        x = nn.functional.leaky_relu(x)
        x = self.conv1(x)
        #print(self.__class__.__name__, 'conv1', x.shape)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        #print(self.__class__.__name__, 'conv2', x.shape)
        x = nn.functional.leaky_relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #print(self.__class__.__name__, 'flatten', x.shape)
        x = self.fc1(x)
        #print(self.__class__.__name__, 'fc1', x.shape)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dims, 784)
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        #print(self.__class__.__name__, 'input', x.shape)
        x = self.fc1(x)
        #print(self.__class__.__name__, 'fc1', x.shape)
        x = nn.functional.leaky_relu(x)
        x = torch.unflatten(x, 1, (16, 7, 7))
        #print(self.__class__.__name__, 'Unflatten', x.shape)
        x = self.conv1(x)
        #print(self.__class__.__name__, 'conv1', x.shape)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        #print(self.__class__.__name__, 'conv2', x.shape)
        x = nn.functional.leaky_relu(x)
        x = self.conv3(x)
        #print(self.__class__.__name__, 'conv3', x.shape)
        return torch.sigmoid(x)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def get_data_loaders():
    global args

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def get_device():
    global args

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        return torch.device("cuda")
    if use_mps:
        return torch.device("mps")
    return torch.device("cpu")


def train_autoencoder(autoencoder, training_data, test_data=None):
    global args

    device = get_device()

    n_samples = len(training_data.dataset)
    n_digits = len(str(n_samples))

    autoencoder.train()
    print("Learning Rate:", args.lr)
    opt = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    #scheduler = StepLR(opt, step_size=1, gamma=args.gamma)
    for epoch in range(args.epochs):
        for batch_idx, (x, y) in enumerate(training_data):
            #x = torch.randn((args.batch_size, 128))
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()

            if batch_idx % args.log_interval == 0:
                used_samples = batch_idx * x.shape[0]
                avg_loss = loss.item() / x.shape[0]
                print(f"Train Epoch: {epoch}",
                      f"[{used_samples:{n_digits}d}/{n_samples:{n_digits}d} ({used_samples/n_samples:5.1%})]",
                      f"\tLast Batch Loss: {avg_loss:.6f}")
                if args.dry_run:
                    break

        if test_data is not None:
            print("")
            show_samples(autoencoder, test_data, epoch, n_samples=10, n_batches=3)
            print("")
            autoencoder.train()

    return autoencoder


def show_samples(autoencoder, test_data, epoch, n_samples, n_batches=1) -> None:
    global args

    device = get_device()

    autoencoder.eval()

    for batch_idx, (x, y) in enumerate(test_data):
        if batch_idx == n_batches:
            break

        x = x.to(device)
        x_hat = autoencoder(x)
        loss = ((x - x_hat) ** 2).sum()

        avg_loss = loss.item()/x.shape[0]

        print(f"Test Set. Batch: {batch_idx}. Loss: {avg_loss:.6f}")

        x = x.cpu()
        x_hat = x_hat.cpu().detach()
        diff = x - x_hat
        plt.figure(figsize=(16, 9))
        plt.suptitle("Top row: Original. Middle: Reconstruction. Bottom: Difference.\n"
                     f"Epoch {epoch}. Batch: {batch_idx}. Loss: {avg_loss:.1f}")
        for i in range(n_samples):
            plt.subplot(3, n_samples, i+1)
            plt.imshow(x[i, 0, :, :], cmap='gray')
            if i == 0:
                plt.ylabel('Original')

            plt.subplot(3, n_samples, i+1+n_samples)
            plt.imshow(x_hat[i, 0, :, :], cmap='gray')
            if i == 0:
                plt.ylabel('Reconstruction')

            plt.subplot(3, n_samples, i+1+(2*n_samples))
            plt.imshow(diff[i, 0, :, :])
            if i == 0:
                plt.ylabel('Difference')

        plt.savefig(os.path.join('out', f'autoencoder_test_res_epoch={epoch:02d}_batch={batch_idx:03d}.png'))
        plt.close()


def main():
    global args

    print("Arguments:")
    for key, val in vars(args).items():
        print(f"- {key}: {val}")
    print("")

    device = get_device()
    training_data, test_data = get_data_loaders()

    autoencoder = AutoEncoder(args.latent_dim).to(device)

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        autoencoder.load_state_dict(torch.load(args.pretrained_model, weights_only=True))
        show_samples(autoencoder, test_data, epoch='Final', n_samples=10, n_batches=1)
    else:
        autoencoder = train_autoencoder(autoencoder, training_data, test_data)
        if args.save_model:
            torch.save(autoencoder.state_dict(),
                       f"auto_encoder_{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}.pt")




if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST AutoEncoder Example')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pretrained-model', type=str, required=False,
                        help='Path to the a .pt file with the pre-trained model weights.')

    args = parser.parse_args()

    main()

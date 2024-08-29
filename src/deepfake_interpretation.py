"""Train deepfake detector on SytleGAN deepfakes."""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image
from tqdm import tqdm

from mnist_integrated import integrate_gradients
from util import WelfordEstimator, get_label, load_folder


def load_image(path_to_file: Path) -> np.ndarray:
    """Load image from path."""
    image = Image.open(path_to_file)
    array = np.nan_to_num(np.array(image), posinf=255, neginf=0)
    return array


class CNN(nn.Module):
    """A simple CNN model."""

    def __init__(self):
        """Create a convolutional neural network."""
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(
            4608, 1024
        )  # Adjust the size according to the input dimensions
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        """Forward step."""
        x = f.relu(self.conv1(x))
        x = f.avg_pool2d(x, kernel_size=2, stride=2)
        x = f.relu(self.conv2(x))
        x = f.avg_pool2d(x, kernel_size=2, stride=2)
        x = f.relu(self.conv3(x))
        x = f.avg_pool2d(x, kernel_size=2, stride=2)
        x = f.relu(self.conv4(x))
        x = f.avg_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # flatten
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Dense(nn.Module):
    """A simple Dense model."""

    def __init__(self):
        """Create a dense linear classifier."""
        super().__init__()
        self.dense = nn.Linear(49152, 2)

    def forward(self, x):
        """Forward step."""
        x = x.reshape((x.shape[0], -1))
        x = self.dense(x)
        return x


def compute_metrics(loss_fun, logits, labels):
    """Compute metrics after training step."""
    loss = loss_fun(logits, labels)
    accuracy = torch.mean((torch.argmax(logits, -1) == labels).type(torch.float32))
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


@torch.no_grad()
def eval_step(net, loss, img, labels):
    """Make one eval step."""
    logits = net(img)
    return compute_metrics(loss, logits=logits, labels=labels)


def transform(image_data):
    """Transform image data."""
    # TODO: Implement the function given in the readme
    return np.zeros_like(image_data)


if __name__ == "__main__":
    ffhq_train, ffhq_val, ffhq_test = load_folder(
        Path("./data/ffhq_style_gan/source_data/A_ffhq"), 67_000, 1_000, 2_000
    )
    gan_train, gan_val, gan_test = load_folder(
        Path("./data/ffhq_style_gan/source_data/B_stylegan"), 67_000, 1_000, 2_000
    )

    fft = True
    epochs = 5
    batch_size = 500

    train = np.concatenate((ffhq_train, gan_train))
    val = np.concatenate((ffhq_val, gan_val))
    np.random.seed(42)
    rng = torch.manual_seed(42)

    np.random.shuffle(train)
    np.random.shuffle(val)

    train_batches = np.array_split(train, len(train) // batch_size)[:50]

    # net = CNN().cuda()
    net = Dense().cuda()
    opt = torch.optim.Adam(net.parameters())
    loss_fun = torch.nn.CrossEntropyLoss()

    estimator = WelfordEstimator()
    image_batch_list = []
    label_batch_list = []

    with Pool(5) as p:
        for path_batch in tqdm(train_batches, "computing training mean and std"):
            loaded = np.stack(p.map(load_image, path_batch))
            image_stack = np.stack(loaded)
            image_batch_list.append(image_stack)
            label_batch_list.append(
                np.array([get_label(path, True) for path in path_batch])
            )
            if fft:
                transform_batch = transform(image_stack)
                estimator.update(transform_batch)
            else:
                estimator.update(image_stack)
    train_mean, train_std = estimator.finalize()
    train_mean, train_std = train_mean.astype(np.float32), train_std.astype(np.float32)
    train_mean, train_std = np.array(train_mean), np.array(train_std)

    print("mean: {}, std: {}".format(train_mean, train_std))

    val_image = np.stack(list(map(load_image, val)))
    if fft:
        transform_val = transform(val_image)
        val_image = transform_val
    val_label_np = np.array([get_label(path, True) for path in val])
    val_image = torch.tensor((val_image - train_mean) / train_std).type(torch.float32)
    val_image = torch.permute(val_image, [0, -1, 1, 2]).cuda()
    val_label = torch.tensor(val_label_np).type(torch.long).cuda()

    for e in range(epochs):
        metrics = eval_step(net, loss_fun, val_image, val_label)
        print(
            "val  , epoch {}, loss {:3.3f}, acc {:3.3f}".format(
                e, metrics["loss"], metrics["accuracy"]
            )
        )

        progress_bar = tqdm(
            zip(image_batch_list, label_batch_list), total=len(image_batch_list)
        )
        for img_batch, label_batch in progress_bar:
            if fft:
                img_batch = transform(img_batch)
            img_batch = (img_batch - train_mean) / train_std
            img_batch = np.transpose(img_batch, [0, -1, 1, 2])
            img_batch = torch.tensor(img_batch).type(torch.float32).cuda()
            label_batch = torch.tensor(label_batch).type(torch.long).cuda()

            # state, metrics = train_step(state, train_ds)
            logits = net(img_batch)
            cost_val = loss_fun(logits, label_batch)
            cost_val.backward()
            opt.step()

            metrics = compute_metrics(loss_fun, logits, label_batch)

            opt.zero_grad()

            progress_bar.set_description(
                "Training. Loss: {:3.3f}, Acc: {:3.3f}".format(
                    metrics["loss"], metrics["accuracy"]
                )
            )

    metrics = eval_step(net, loss_fun, val_image, val_label)
    print(
        "val  , epoch {}, loss {:3.3f}, acc {:3.3f}".format(
            e, metrics["loss"], metrics["accuracy"]
        )
    )

    # test metrics
    test = np.concatenate((ffhq_test, gan_test))
    # load test data
    test_image = np.stack(list(map(load_image, test)))
    if fft:
        test_image = transform(test_image)
    test_label_np = np.array([get_label(path, True) for path in test])

    test_image = torch.tensor((test_image - train_mean) / train_std).type(torch.float32)
    test_image = torch.permute(test_image, [0, -1, 1, 2]).cuda()
    test_label = torch.tensor(test_label_np).type(torch.long).cuda()
    # get the
    test_metrics = eval_step(net, loss_fun, test_image, test_label)
    print(
        "test, loss {:3.3f}, acc {:3.3f}".format(
            test_metrics["loss"], test_metrics["accuracy"]
        )
    )

    # visualize the linear network.
    if type(net) is Dense:
        import matplotlib.pyplot as plt

        stacked_ffhq_val = np.stack(list(map(load_image, ffhq_val)))
        fft_ffhq_val = transform(stacked_ffhq_val)
        stacked_gan_val = np.stack(list(map(load_image, gan_val)))
        fft_gan_val = transform(stacked_gan_val)

        fft_ffhq_val = np.mean(fft_ffhq_val, (0, -1))
        fft_gan_val = np.mean(fft_gan_val, (0, -1))

        diff = np.abs(fft_ffhq_val - fft_gan_val)
        plt.subplot(1, 2, 1)
        plt.title("Real mean-log fft2")
        plt.imshow(fft_ffhq_val, vmin=np.min(fft_ffhq_val), vmax=np.max(fft_ffhq_val))
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("Fake mean-log fft2")
        plt.imshow(fft_gan_val, vmin=np.min(fft_ffhq_val), vmax=np.max(fft_ffhq_val))
        plt.colorbar()
        plt.savefig("real_fake_mean-log_fft2.jpg")

        plt.subplots()
        plt.title("Row averaged shifted mean-log fft2")
        plt.plot(np.fft.fftshift(np.mean(fft_ffhq_val, 0))[64:], ".", label="real")
        plt.plot(np.fft.fftshift(np.mean(fft_gan_val, 0))[64:], ".", label="fake")
        plt.xlabel("frequency")
        plt.ylabel("magnitude")
        plt.legend()
        plt.savefig("row_average_shifted_mean-log_fft2.jpg")

        plt.subplots()
        plt.title("Mean frequency difference")
        plt.imshow(diff)
        plt.colorbar()
        plt.savefig("mean_freq_difference.jpg")

        # TODO: Visualize the weight array `net.dense.weight`.
        # By reshaping and plotting the weight matrix.

    if type(net) is CNN:
        import matplotlib.pyplot as plt

        ig_out = integrate_gradients(net=net, test_images=test_image, output_digit=1)
        plt.imshow(np.mean(ig_out, -1))

        plt.savefig("integrated_gradients.jpg")

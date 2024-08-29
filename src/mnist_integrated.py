"""Identify mnist digits."""

import argparse
import struct
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import grad
from tqdm import tqdm

from input_opt import CNN


def get_mnist_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Return the mnist test data set in numpy arrays.

    Returns:
        (array, array): A touple containing the test
        images and labels.
    """
    with open("./data/MNIST/t10k-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_test = data.reshape((size, nrows, ncols))

    with open("./data/MNIST/t10k-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_test = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    # if gpu:
    #    return cp.array(img_data_test), cp.array(lbl_data_test)
    return img_data_test, lbl_data_test


def get_mnist_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the mnist training data set.

    Returns:
        (array, array): A touple containing the training
        images and labels.
    """
    with open("./data/MNIST/train-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_train = data.reshape((size, nrows, ncols))

    with open("./data/MNIST/train-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    # if gpu:
    #    return cp.array(img_data_train), cp.array(lbl_data_train)
    return img_data_train, lbl_data_train


def normalize(
    data: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """Normalize the input array.

    After normalization the input
    distribution should be approximately standard normal.

    Args:
        data (np.array): The input array.
        mean (float): Data mean, re-computed if None.
            Defaults to None.
        std (float): Data standard deviation,
            re-computed if None. Defaults to None.

    Returns:
        np.array, float, float: Normalized data, mean and std.
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / std, mean, std


def cross_entropy(label: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute the cross entropy of one-hot encoded labels and the network output.

    Args:
        label (torch.Tensor): The image labels of shape [batch_size, 10].
        out (torch.Tensor): The network output of shape [batch_size, 10].

    Returns:
        torch.Tensor, The loss scalar.

    """
    left = -label * torch.log(out + 1e-8)
    right = -(1 - label) * torch.log(1 - out + 1e-8)
    return torch.mean(left + right)


def get_acc(cnn, img_data, label_data):
    """Compute the network accuracy."""
    img_data = torch.tensor(img_data).type(torch.float32).cuda()
    label_data = torch.tensor(label_data).cuda()
    out = cnn(torch.unsqueeze(img_data, 1))
    rec = torch.argmax(out, axis=1)
    acc = torch.sum((rec == label_data).type(torch.float32)) / len(label_data)
    return acc


@torch.no_grad()
def integrate_gradients(net, test_images, output_digit, steps_m=300):
    """Calculate integrated gradients."""
    g_list = []
    for test_image_x in tqdm(test_images, desc="Integrating Gradients"):
        # TODO: create a list for the gradients.
        # step_g_list = []
        # TODO: create a black reference image using jnp.zeros_like .
        # black_image_x_prime = jnp.zeros_like(test_image_x)
        # TODO: Loop over the integration steps.
        # for current_step_k in range(steps_m):
        #  # TODO: compute the input to F from equation 5 in the slides.
        #  current_point = black_image_x_prime + (current_step_k/steps_m) * (test_image_x - black_image_x_prime)
        #  # TODO: define a forward pass for jax.grad
        #  def eval_fun(point):
        #    logits = net.apply(weights, point)
        #    return logits[0, output_digit]
        #  # TODO: use jax.grad to find the gradient with repsect to the input image.
        #  grad_fun = jaxday_14_exercise_interpretability_solution/scripts/train.slurm.grad(eval_fun)
        #  grads = grad_fun(jnp.expand_dims(current_point, 0))
        #  # TODO: append the gradient to yout list
        #  step_g_list.append(grads)
        # TODO: Return the sum of the of the list elements.
        # g_sum = jnp.sum(jnp.concatenate(step_g_list), 0)

        test_image_x = torch.unsqueeze(torch.unsqueeze(test_image_x, 0), 0)
        black_image_x_prime = torch.zeros_like(test_image_x)
        scales = (
            (torch.arange(steps_m) / steps_m).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        points = black_image_x_prime + scales * (test_image_x - black_image_x_prime)

        def eval_fun(point):
            logits = net(point)
            return torch.sum(logits[:, output_digit], 0)

        grad_fun = grad(eval_fun)
        # points = points.transpose(1, -1)
        grads = grad_fun(points)
        g_sum = torch.sum(grads, 0)
        integrated_g = (test_image_x - black_image_x_prime) * g_sum * (1.0 / steps_m)
        g_list.append(integrated_g)
        net.zero_grad()
    return torch.mean(torch.cat(g_list, 0), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Networks on MNIST.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    args = parser.parse_args()
    print(args)

    rng = np.random.default_rng(seed=42)

    batch_size = 50
    val_size = 500
    epochs = 20
    img_data_train, lbl_data_train = get_mnist_train_data()
    img_data_val, lbl_data_val = img_data_train[:val_size], lbl_data_train[:val_size]
    img_data_train, lbl_data_train = (
        img_data_train[val_size:],
        lbl_data_train[val_size:],
    )
    img_data_train, mean, std = normalize(img_data_train)
    img_data_val, _, _ = normalize(img_data_val, mean, std)

    my_file = Path("weights.pth")
    if not my_file.is_file():
        cnn = CNN().cuda()
        opt = optim.Adam(lr=args.lr, params=cnn.parameters())
        cost_fun = nn.CrossEntropyLoss()

        for e in range(epochs):
            shuffled = rng.permutation(len(img_data_train))
            img_data_train = img_data_train[shuffled]
            lbl_data_train = lbl_data_train[shuffled]

            img_batches = np.split(
                img_data_train, img_data_train.shape[0] // batch_size, axis=0
            )
            label_batches = np.split(
                lbl_data_train, lbl_data_train.shape[0] // batch_size, axis=0
            )

            bar = tqdm(
                zip(img_batches, label_batches),
                total=len(img_batches),
                desc="Training CNN",
            )

            for img_batch, label_batch in bar:
                img_batch, label_batch = (
                    torch.tensor(np_array).cuda()
                    for np_array in (img_batch, label_batch)
                )
                img_batch = img_batch.type(torch.float32)
                # cel = cross_entropy(nn.one_hot(label_batches[no], num_classes=10),
                #                    out)
                y_hat = cnn(torch.unsqueeze(img_batch, 1))
                ce_val = cost_fun(y_hat, label_batch)
                ce_val.backward()
                opt.step()
                opt.zero_grad()

                bar.set_description("Loss: {:2.3f}".format(ce_val.item()))
            print("Epoch: {}, loss: {}".format(e, ce_val.item()))

            # train_acc = get_acc(img_data_train, lbl_data_train)
            val_acc = get_acc(cnn, img_data_val, lbl_data_val)
            print("Validation accuracy: {:3.3f}".format(val_acc))

        torch.save(cnn.state_dict(), "weights.pth")

    else:
        cnn_weights = torch.load(open("weights.pth", "rb"))
        cnn = CNN()
        cnn.load_state_dict(cnn_weights)
        cnn.cuda()

    print("Training done. Testing...")
    img_data_test, lbl_data_test = get_mnist_test_data()
    img_data_test, mean, std = normalize(img_data_test, mean, std)
    test_acc = get_acc(cnn, img_data_test, lbl_data_test)
    print("Done. Test acc: {}".format(test_acc))

    print("IG for digit 0")
    cnn = cnn.cpu()
    ig_0 = integrate_gradients(
        cnn,
        torch.tensor(img_data_test[:400]).type(torch.float32),
        output_digit=0,
        steps_m=300,
    )
    print("IG for digit 1")
    plt.imshow(ig_0[0])
    plt.savefig("integrated_gradients_0.jpg")
    ig_1 = integrate_gradients(
        cnn,
        torch.tensor(img_data_test[:400]).type(torch.float32),
        output_digit=1,
        steps_m=300,
    )
    plt.imshow(ig_1[0])
    plt.savefig("integrated_gradients_1.jpg")
    print("stop")

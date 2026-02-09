# Exercise: Interpretable machine learning

# Task 1: Input optimization.
Open `src/input_opt.py`. In this exercise we will turn the network optimization problem around. Instead of updating weights to minimize loss, we will keep the weights fixed and update the input image to maximize the activation of a specific output neuron.

The network `./data/weights.pth` contains network weights pre-trained on MNIST. We want to generate an image $\mathbf{x}$ that the network strongly believes shows a specific digit.

Mathematically, we want to maximize:
```math
\max_\mathbf{x} y_i = f(\mathbf{x}, \theta) .
```
where $f$ is the neural network function, $\mathbf{x}$ is the input image, $\theta$ are the fixed weights of the network, and $y_i$ is the output of the target neuron corresponding to the digit we want to visualize.


1. Complete `forward_pass`: Implement the function to return the scalar output of the target neuron.

The gradients are computed using `torch.func.grad`. Start with a network input image $\mathbf{x}$ of shape `[1, 1, 28, 28]`.

2. Write an optimization loop to iteratively update the input image $\mathbf{x}$ based on the computed gradients.

3. Compare and visualize the results of starting with a random noise image versus starting with a image filled with ones.

# Task 2 Integrated Gradients (Optional):


Reuse your MNIST digit recognition code. Implement IG as discussed in the lecture. Recall the equation

```math
\text{IntegratedGrads}_i(x) = (x_i - x_i') \cdot \frac{1}{m} \sum_{k=1}^m \frac{\partial F (x' + \frac{k}{m} \cdot (x - x'))}{\partial x_i}.
```

$\frac{\partial F}{\partial x_i}$ denotes the gradients with respect to the input color-channels $i$.
$x'$ denotes a baseline black image. And $x$ symbolizes an input we are interested in.
Finally, $m$ denotes the number of summation steps from the black baseline image to the interesting input.

Follow the todos in `./src/mnist_integrated.py` and then run `scripts/integrated_gradients.slurm`.



# Task 3 Deepfake detection (Optional):
In this exercise we will consider 128 by 128-pixel fake images from [StyleGAN](https://github.com/NVlabs/stylegan) and pictures of real people from the  [Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset) dataset.

Flickr-Faces-HQ images depict real people, such as the person below:

![real person](./figures/real.png)

Generative adversarial networks allow the generation of fake images at scale. Does the picture below seem real? 

![fake person](./figures/fake.png)

How can we identify the fake? Given that modern neural networks can generate hundreds of fake images per second can we create a classifier to automate the process?

### 3.1 Getting started:
1. Move to the `data` folder in your terminal. Download [ffhq_style_gan.zip](https://drive.google.com/uc?id=1MOHKuEVqURfCKAN9dwp1o2tuR19OTQCF&export=download) on bender using the command
   ```bash
   gdown https://drive.google.com/uc?id=1MOHKuEVqURfCKAN9dwp1o2tuR19OTQCF
   ```
   If `gdown` is not installed, type `pip install gdown` and then try again.
2. Type `export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE` to make unzipping big archives possible.
3. Extract the image pairs here by executing `unzip ffhq_style_gan.zip` in the terminal.

The desired outcome is to have a folder called `ffhq_style_gan` in the project data-folder.


### 3.2 Analyzing the data
The `load_folder` function from the `util` module loads both real and fake data.
Code to load the data is already present in the `deepfake_interpretation.py` file.

1. Implement the `transform` function to compute log-scaled frequency domain representations of samples from both sources via

   ``` math
   \mathbf{F}_I =  \log_e (| \mathcal{F}_{2d}(\mathbf(I)) | + \epsilon ), \text{ with } \mathbf{I} \in \mathbb{R}^{h,w,c}, \epsilon \approx 0 .
   ```

   Above `h`, `w` and `c` denote image height, width and columns. `Log` denotes the natural logarithm, and bars denote the absolute value. A small epsilon is added for numerical stability.

   Use the numpy functions `np.log`, `np.abs`, `np.fft.fft2`. By default, `fft2` transforms the last two axes. The last axis contains the color channels in this case. We are looking to transform the rows and columns.

2. Plot mean spectra for real and fake images as well as their difference over the entire validation or test sets. For that run the script `scripts/train.slurm`.

3. `scripts/train.slurm` also trains a linear classifier (consisting of a single `nn.Linear`-layer) to distinguish real from fake images on the log-scaled Fourier coefficients. We want to visualize the weights of the trained classifier. For that go to `src/deepfake_interpretation.py` and implement the TODO at the end of the file. What do you see?

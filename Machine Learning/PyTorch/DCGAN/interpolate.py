# Inspired by https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
# example of interpolating between generated faces
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils
import model

# Set random seed for reproducibility
import random
manualSeed = 999 # Try 5989 (Pants to shirt?)
manualSeed = random.randint(1, 10000) # use if you want new results
print(manualSeed)
#random.seed(manualSeed)
torch.manual_seed(manualSeed)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	#x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	#z_input = x_input.reshape(latent_dim, n_samples, )
	#return z_input
    return torch.randn(n_samples, model.nz, 1, 1, device=device)

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = torch.linspace(0, 1, steps=n_steps)
    # linear interpolate vectors

    size = [len(ratios)]+list(p1.shape)
    vectors = torch.zeros(*size)
    #vectors = list()
    for i in range(len(ratios)):
    	ratio = ratios[i]; v = (1.0 - ratio) * p1 + ratio * p2;vectors[i]=v
        #vectors.append(v)
    return vectors

def plot_generated(X):
    # Plot the fake images from the last epoch
    plt.figure(figsize=(10,5))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(X,(1,2,0)))
    plt.show()


ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
model.nc = 3
dcgan = model.DCGAN(ngpu, device)
dcgan.load(
    #'checkpoints/2021-05-25T09-16-41_FashionMNIST/dcgan_generator.chkpt','checkpoints/2021-05-25T09-16-41_FashionMNIST/dcgan_discriminator.chkpt'
    'checkpoints/2021-05-25T00-33-43_CelebFaces/dcgan_generator.chkpt', 'checkpoints/2021-05-25T00-33-43_CelebFaces/dcgan_discriminator.chkpt'
)

# generate points in latent space
pts = generate_latent_points(model.nz, 2)
# interpolate points in latent space
interpolated = interpolate_points(pts[0], pts[1])
# generate images
noise = torch.randn(100, model.nz, 1, 1, device=device)
noise = interpolated
X = dcgan.generate(noise)
# plot the result
plot_generated(vutils.make_grid(X, padding=2, normalize=True))#len(interpolated))


### Vector arithmetic
noise = torch.randn(100, model.nz, 1, 1, device=device)
X = dcgan.generate(noise)
plot_generated(vutils.make_grid(X, padding=2, normalize=True))

def empty_tensor(X, width):
    size = [width]+list(X[0].shape)
    tensor = torch.zeros(*size)
    return tensor

new_noise = empty_tensor(noise,1)
a, b, c = 0, 20, 90
new_noise[0] = noise[a] - noise[b] + noise[c]
new_img = dcgan.generate(new_noise)

size = [4]+list(X[0].shape)
vectors = torch.zeros(*size)
vectors[0] = X[a]
vectors[1] = X[b]
vectors[2] = X[c]
vectors[3] = new_img[0]
plot_generated(vutils.make_grid(vectors, padding=2, normalize=True))

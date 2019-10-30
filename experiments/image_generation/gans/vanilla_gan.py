import os

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn as nn
from experiments.image_generation.utils import Logger



# def mnist_data():
# 	# , transforms.Normalize((.5, .5, .5), (.5,.5,.5))
# 	compose = transforms.Compose([transforms.ToTensor()])
# 	out_dir = './dataset'
# 	return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
#
#
# # Load data
# data = mnist_data()
# # Create loader with data, so that we can iterate over it
# data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# # Num batches
# num_batches = len(data_loader)


data_transform = transforms.Compose([
	# transforms.RandomResizedCrop(224),
	# transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
hymenoptera_dataset = datasets.ImageFolder(root='dataset/extra_data', transform=data_transform)
print(len(hymenoptera_dataset))
data_loader = torch.utils.data.DataLoader(hymenoptera_dataset, batch_size=100, shuffle=True, num_workers=4)
# Num batches
num_batches = len(data_loader)

# n_features = 784
# n_channels = 1
# width = 28
# height = 28
# n_out = 1

n_features = 3 * 64 * 64
n_channels = 3
width = 64
height = 64
n_out = 1
n_noise = 100


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
D_PATH = '../models/vgan/anime_d.pth'
G_PATH = '../models/vgan/anime_g.pth'


class DiscriminatorNet(torch.nn.Module):
	"""
	A three hidden-layer discriminative neural network
	"""

	def __init__(self):
		super(DiscriminatorNet, self).__init__()
		# n_features = 784
		# n_out = 1

		self.hidden0 = nn.Sequential(
			nn.Linear(n_features, 1024),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		self.hidden1 = nn.Sequential(
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		self.hidden2 = nn.Sequential(
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		self.out = nn.Sequential(
			torch.nn.Linear(256, n_out),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		x = self.hidden0(x)
		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.out(x)
		return x


discriminator = DiscriminatorNet()

# load model
if os.path.isfile(D_PATH):
	discriminator.load_state_dict(torch.load(D_PATH))
discriminator.to(device)


def images_to_vectors(images):
	return images.view(images.size(0), n_features)


def vectors_to_images(vectors):
	return vectors.view(vectors.size(0), n_channels, width, height)


class GeneratorNet(torch.nn.Module):
	"""
	A three hidden-layer generative neural network
	"""

	def __init__(self):
		super(GeneratorNet, self).__init__()
		# n_features = 100
		# n_out = 784

		self.hidden0 = nn.Sequential(
			nn.Linear(n_noise, 256),
			nn.LeakyReLU(0.2)
		)
		self.hidden1 = nn.Sequential(
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2)
		)
		self.hidden2 = nn.Sequential(
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2)
		)

		self.out = nn.Sequential(
			nn.Linear(1024, n_features),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.hidden0(x)
		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.out(x)
		return x


generator = GeneratorNet()

if os.path.isfile(G_PATH):
	generator.load_state_dict(torch.load(G_PATH))

generator.to(device)


def noise(size):
	"""
	Generates a 1-d vector of gaussian sampled random values
	"""
	# n = Variable(torch.randn(size, 100))
	n = torch.randn((size, 100), requires_grad=True).to(device)
	return n


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()


def ones_target(size):
	"""Tensor containing ones, with shape = size"""
	# data = Variable(torch.ones(size, 1))
	data = torch.ones((size, 1)).to(device)
	return data


def zeros_target(size):
	"""
	Tensor containing zeros, with shape = size
	"""
	# data = Variable(torch.zeros(size, 1))
	data = torch.zeros((size, 1)).to(device)
	return data


def train_discriminator(optimizer, real_data, fake_data):
	N = real_data.size(0)
	# Reset gradients
	optimizer.zero_grad()

	# 1.1 Train on Real Data
	prediction_real = discriminator(real_data)
	# Calculate error and backpropagate
	error_real = loss(prediction_real, ones_target(N))
	error_real.backward()

	# 1.2 Train on Fake Data
	prediction_fake = discriminator(fake_data)
	# Calculate error and backpropagate
	error_fake = loss(prediction_fake, zeros_target(N))
	error_fake.backward()

	# 1.3 Update weights with gradients
	optimizer.step()

	# Return error and predictions for real and fake inputs
	return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
	N = fake_data.size(0)
	# Reset gradients
	optimizer.zero_grad()
	# Sample noise and generate fake data
	prediction = discriminator(fake_data)
	# Calculate error and backpropagate
	error = loss(prediction, ones_target(N))
	error.backward()
	# Update weights with gradients
	optimizer.step()
	# Return error
	return error


num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='ANIME')

# Total number of epochs to train
num_epochs = 2000
for epoch in range(num_epochs):
	for n_batch, samples in enumerate(data_loader):
		(real_batch, _) = samples
		real_batch = real_batch.to(device)
		N = real_batch.size(0)

		# 1. Train Discriminator
		real_data = images_to_vectors(real_batch)

		# Generate fake data and detach (so gradients are not calculated for generator)
		fake_data = generator(noise(N)).detach()

		# Train D
		d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

		# 2. Train Generator
		# Generate fake data
		fake_data = generator(noise(N))

		# Train G
		g_error = train_generator(g_optimizer, fake_data)

		# Log batch error
		logger.log(d_error, g_error, epoch, n_batch, num_batches)

		# Display Progress every few batches
		if n_batch % 100 == 0:
			test_images = vectors_to_images(generator(test_noise))
			test_images = test_images.data
			logger.log_images(test_images.cpu(), num_test_samples, epoch, n_batch, num_batches)

			# Display status Logs
			logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake)


	# save model every 100 epoch
	if epoch % 10 == 0:
		torch.save(discriminator.state_dict(), D_PATH)
		torch.save(generator.state_dict(), G_PATH)

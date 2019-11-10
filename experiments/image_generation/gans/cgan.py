import os

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn as nn
from experiments.image_generation.utils import Logger

#	mnist data
# def mnist_data():
# , transforms.Normalize((.5, .5, .5), (.5,.5,.5))
compose = transforms.Compose([transforms.ToTensor()])
out_dir = './dataset'
data = datasets.CIFAR10(root=out_dir, train=True, transform=compose, download=True)

# Load data
# data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

# data_transform = transforms.Compose([
# 	# transforms.RandomResizedCrop(224),
# 	# transforms.RandomHorizontalFlip(),
# 	transforms.ToTensor(),
# 	# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# hymenoptera_dataset = datasets.ImageFolder(root='../dataset/extra_data', transform=data_transform)
# print(len(hymenoptera_dataset))
# data_loader = torch.utils.data.DataLoader(hymenoptera_dataset, batch_size=128, shuffle=True, num_workers=4)
# # Num batches
# num_batches = len(data_loader)


n_features = 3 * 64 * 64
n_channels = 3
width = 64
height = 64
n_out = 1
n_noise = 100
clip = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
D_PATH = 'models/vgan/anime_d.pth'
G_PATH = 'models/vgan/anime_g.pth'


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


class DiscriminatorNet(nn.Module):
	"""
	A three hidden-layer discriminative neural network
	"""

	def __init__(self):
		super(DiscriminatorNet, self).__init__()
		# n_features = 784
		# n_out = 1

		self.hidden0 = nn.Sequential(
			nn.Conv2d(3, 64, 3, stride=2, padding=1, ),
			nn.LeakyReLU(0.2),
			# nn.MaxPool2d(2, 2)
		)

		self.hidden1 = nn.Sequential(
			nn.Conv2d(64, 128, 3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			# nn.MaxPool2d(2, 2)
		)

		self.hidden2 = nn.Sequential(
			nn.Conv2d(128, 256, 3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			# nn.MaxPool2d(2, 2, ceil_mode=True)
		)

		# self.hidden3 = nn.Sequential(
		# 	nn.Conv2d(256, 512, 3, stride=2, padding=1),
		# 	nn.BatchNorm2d(512),
		# 	nn.LeakyReLU(0.2),
		# 	# nn.MaxPool2d(2, 2, ceil_mode=True)
		# )

		self.out = nn.Sequential(
			# nn.Conv2d(256, 1, 4, padding=0),
			nn.Linear(256 * 4 * 4 + 10, n_out),
			# nn.Conv2d(512, 1, 5, stride=2, padding=1),
			# nn.BatchNorm1d(n_out),

			# in wgan, should not use sigmoid
			nn.Sigmoid()
		)

	def forward(self, x, y):
		"""
		input 3*64*64
		:param x:
		:return:
		"""
		x = self.hidden0(x)
		x = self.hidden1(x)
		x = self.hidden2(x)
		# x = self.hidden3(x)

		x = x.view(-1, 256 * 4 * 4)
		x = torch.cat((x, y), 1)

		x = self.out(x)
		return x


discriminator = DiscriminatorNet()

# # load model
# if os.path.isfile(D_PATH):
# 	discriminator.load_state_dict(torch.load(D_PATH))
discriminator.to(device)
discriminator.apply(weights_init)


def images_to_vectors(images):
	return images.view(images.size(0), n_features)


def vectors_to_images(vectors):
	return vectors.view(vectors.size(0), n_channels, width, height)


class GeneratorNet(nn.Module):
	"""
	A three hidden-layer generative neural network
	"""

	def __init__(self):
		super(GeneratorNet, self).__init__()
		# n_features = 100
		# n_out = 784

		# self.hidden0 = nn.Sequential(
		# 	nn.Linear(n_noise, 4 * 4 * 1024),
		# 	nn.BatchNorm1d(4 * 4 * 1024),
		# 	nn.ReLU()
		# )

		self.hidden0 = nn.Sequential(
			nn.ConvTranspose2d(110, 512, 4, 1, 0),
			nn.BatchNorm2d(512),
			nn.ReLU()
		)

		self.hidden1 = nn.Sequential(
			nn.ConvTranspose2d(512, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU()
		)

		self.hidden2 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)

		self.out = nn.Sequential(
			nn.ConvTranspose2d(128, 3, 4, 2, 1),
			# nn.BatchNorm2d(64),
			nn.Tanh()
		)

	# self.out = nn.Sequential(
	# 	nn.ConvTranspose2d(64, 3, 4, 2, 1),
	# 	nn.Tanh()
	# )

	def forward(self, x):
		x = x.view(-1, 110, 1, 1)
		x = self.hidden0(x)

		x = self.hidden1(x)
		x = self.hidden2(x)
		# x = self.hidden3(x)
		x = self.out(x)

		return x


generator = GeneratorNet()
#
# if os.path.isfile(G_PATH):
# 	generator.load_state_dict(torch.load(G_PATH))

generator.to(device)
generator.apply(weights_init)


def noise(size):
	"""
	Generates a 1-d vector of gaussian sampled random values
	"""
	# n = Variable(torch.randn(size, 100))
	n = torch.randn((size, 100), requires_grad=True).to(device)
	# n = torch.randn((size, 100, 1, 1), requires_grad=True).to(device)
	# n = torch.randn((size, 100), requires_grad=True).to(device)
	return n


d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0002)


def train_discriminator(optimizer, real_data, fake_data, y):
	N = real_data.size(0)
	# Reset gradients
	optimizer.zero_grad()

	# 1.1 Train on Real Data
	prediction_real = discriminator(real_data, y)
	# Calculate error and backpropagate
	# error_real = loss(prediction_real, ones_target(N))
	# error_real.backward()

	# 1.2 Train on Fake Data
	# fake_data = fake_data.view(-1, 3, 64, 64)
	prediction_fake = discriminator(fake_data, y)
	# Calculate error and backpropagate
	# error_fake = loss(prediction_fake, zeros_target(N))
	# error_fake.backward()

	D_loss = -(torch.mean(prediction_real) - torch.mean(prediction_fake))
	D_loss.backward()

	# 1.3 Update weights with gradients
	optimizer.step()

	# weight(gradient) clipping
	# # torch.clamp_(discriminator.parameters(), min=-clip, max=clip)
	# w = discriminator.weight.data
	# w = w.clamp(-clip, clip)
	# discriminator.weight.data = w

	for p in discriminator.parameters():
		p.data.clamp_(-clip, clip)

	# Return error and predictions for real and fake inputs
	# return error_real + error_fake, prediction_real, prediction_fake
	return D_loss, prediction_real, prediction_fake


def train_generator(optimizer, fake_data, y):
	N = fake_data.size(0)

	# Reset gradients
	optimizer.zero_grad()

	# Sample noise and generate fake data
	prediction = discriminator(fake_data, y)

	# Calculate error and backpropagate
	# error = loss(prediction, ones_target(N))
	G_loss = -torch.mean(prediction)

	G_loss.backward()

	# Update weights with gradients
	optimizer.step()

	# Return error
	return G_loss


num_test_samples = 16
test_noise = noise(num_test_samples)
test_condition = torch.randint(10, (num_test_samples,)).to(device)
test_labels = torch.zeros(num_test_samples, 10).to(device)
test_labels[torch.arange(num_test_samples), test_condition] = 1

test_z_y = torch.cat((test_noise, test_labels), 1).to(device)

# Create logger instance
logger = Logger(model_name='cGAN', data_name='cifar')

# Total number of epochs to train
num_epochs = 2000
for epoch in range(num_epochs):
	for n_batch, samples in enumerate(data_loader):
		(real_batch, real_labels) = samples
		real_batch = real_batch.to(device)
		real_labels = real_labels.to(device)
		N = real_batch.size(0)

		# 0. change to one-hot encoding
		one_hot = torch.zeros(N, 10).to(device)
		one_hot[torch.arange(N).to(device), real_labels] = 1
		# one_hot.to(device)

		# 1. Train Discriminator
		# real_data = images_to_vectors(real_batch)

		# Generate fake data and detach (so gradients are not calculated for generator)
		z = noise(N)
		z_y = torch.cat((z, one_hot), 1).to(device)
		y = one_hot
		fake_data = generator(z_y).detach()

		# Train D
		d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_batch, fake_data, y)

		# 2. Train Generator
		# Generate fake data
		fake_data = generator(z_y)

		# Train G
		g_error = train_generator(g_optimizer, fake_data, y)

		# Log batch error
		logger.log(d_error, g_error, epoch, n_batch, num_batches)

		# Display Progress every few batches
		if n_batch % 100 == 0:
			# test_images = vectors_to_images(generator(test_z_y))
			test_images = generator(test_z_y).data
			logger.log_images(test_images.cpu(), num_test_samples, epoch, n_batch, num_batches)

			# Display status Logs
			logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake)
#
# # save model every 100 epoch
# if epoch % 10 == 0:
# 	torch.save(discriminator.state_dict(), D_PATH)
# 	torch.save(generator.state_dict(), G_PATH)

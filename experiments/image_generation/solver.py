from experiments.image_generation.models import Generator, Discriminator


class Solver(object):

	def __init__(self):
		self.generator = Generator()
		self.discriminator = Discriminator()
		return

	def update_discriminator(self, training_data, generator):
		real_images = sample_batch_data(training_data, batch_size)
		noise = sample_batch_noise(batch_size, noise_dim)
		fake_images = generator(noise)
		real_predicts = self.discriminator(real_images)
		fake_predicts = self.discriminator(fake_images)
		d_loss = loss_d_fn(real_predicts, real_labels, fake_predicts, fake_labels)
		d_grad = gradients(d_loss, d_params)
		d_params = updates(d_params, d_grad)

	# do not update the parameters of generator
	return

	def update_generator(self, discriminator):
		noise = sample_batch_noise(noise_dim, batch_size)
		fake_images = self.generator(noise)
		fake_predicts = self.discriminator(fake_images)
		g_loss = loss_g_fn(fake_predicts, real_labels)
		g_grad = gradients(g_loss, g_params)
		g_params = updates(g_params, g_grad)
		# do not update the parameters of discriminator
		return

	def train(self):
		max_iteration = 1000
		d_update = 10
		g_update = 10
		for i in range(max_iteration):
			for j in range(d_update):
				self.discriminator = self.update_discriminator(training_data, generator)
			for k in range(g_update):
				self.generator = self.update_generator(self.discriminator)

		return

	def test(self):
		noise = sample_batch_noise(batch_size, noise_dim)
		output_images = self.generator(noise)
		return

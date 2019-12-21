import os

from keras.backend import binary_crossentropy
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

from keras.utils import plot_model


class VAENetwork:
    def __init__(self, image_size) -> None:
        super().__init__()
        self.image_size = image_size
        input_shape = (image_size, image_size, 1)
        self.inputs = Input(shape=input_shape, name='encoder_input')
        self.batch_size = 128
        self.kernel_size = 5
        self.filters = 16
        self.latent_dim = 10
        self.epochs = 30
        self.outputs = None
        self.vae = None
        self.encoder = None
        self.Decoder = None

    def designNetwork(self):
        # network parameters

        # VAE model = encoder + decoder
        # build encoder model
        x = self.inputs
        for i in range(2):
            self.filters *= 2
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

        # shape info needed to build decoder model
        shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(self.inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        plot_model(self.encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(2):
            x = Conv2DTranspose(filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
            self.filters //= 2

        self.outputs = Conv2DTranspose(filters=1,
                                       kernel_size=self.kernel_size,
                                       activation='sigmoid',
                                       padding='same',
                                       name='decoder_output')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, self.outputs, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = Model(self.inputs, outputs, name='vae')

    def trainOrGetTrained(self, x_train, x_test, mse, modelWeightsName):
        # VAE loss = mse_loss or xent_loss + kl_loss
        if mse:
            reconstruction_loss = mse(K.flatten(self.inputs), K.flatten(self.outputs))
        else:
            reconstruction_loss = binary_crossentropy(K.flatten(self.inputs),
                                                      K.flatten(self.outputs))

        reconstruction_loss *= self.image_size * self.image_size
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='rmsprop')
        self.vae.summary()
        plot_model(self.vae, to_file='vae_cnn.png', show_shapes=True)

        if os.path.exists(modelWeightsName):
            self.vae = self.vae.load_weights(modelWeightsName)
        else:
            # train the autoencoder
            self.vae.fit(x_train,
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         validation_data=(x_test, None))
            self.vae.save_weights(modelWeightsName)

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # then z = z_mean + sqrt(var)*eps
    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        self.z_mean, self.z_log_var = args
        batch = K.shape(self.z_mean)[0]
        dim = K.int_shape(self.z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return self.z_mean + K.exp(0.5 * self.z_log_var) * epsilon

    def predictDecoder(self, latentSpace):
        x_decoded = self.decoder.predict(latentSpace)
        return x_decoded[0].reshape(self.image_size, self.image_size)

    def predictEncoder(self, images):
        res = self.encoder.predict(images, batch_size=self.batch_size)
        return res[2]

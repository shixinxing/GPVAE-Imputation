from .gp_kernel import *
from .building_blocks import *
from tensorflow_probability import distributions as tfd
import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(self, latent_dim, data_dim, time_length,
                 encoder_sizes=(64, 64), encoder=DiagonalEncoder,
                 decoder_sizes=(64, 64), decoder=BernoulliDecoder,
                 image_preprocessor=None, beta=1.0, M=1, K=1, **kwargs):
        """ Basic Variational Autoencoder with Standard Normal prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality  # 为784
            :param time_length: time series duration
            # 数据shape为：[batch_size, time_length, data_dim]
            
            :param encoder_sizes: layer sizes for the encoder network
            :param encoder: encoder model class {Diagonal, Joint, BandedJoint}Encoder
            :param decoder_sizes: layer sizes for the decoder network
            :param decoder: decoder model class {Bernoulli, Gaussian}Decoder
            
            :param image_preprocessor: 2d-convolutional network used for image data preprocessing
            :param beta: tradeoff coefficient between reconstruction and KL terms in ELBO
            :param M: number of Monte Carlo samples for ELBO estimation
            :param K: number of importance weights for IWAE model (see: https://arxiv.org/abs/1509.00519)
        """
        super(VAE, self).__init__()
        self.latent_dim, self.data_dim, self.time_length = latent_dim, data_dim, time_length

        self.encoder = encoder(latent_dim, encoder_sizes, **kwargs)
        self.decoder = decoder(data_dim, decoder_sizes)
        self.preprocessor = image_preprocessor
        self.prior = None   # 自己补上的

        self.beta = beta
        self.K = K
        self.M = M

    def encode(self, x):
        x = tf.identity(x)  # in case x is not a Tensor already...   # [1,10,784]
        if self.preprocessor is not None:
            x_shape = x.shape.as_list()
            new_shape = [x_shape[0] * x_shape[1]] + list(self.preprocessor.image_shape)  # [10,28,28,1]
            x_reshaped = tf.reshape(x, new_shape)
            x_preprocessed = self.preprocessor(x_reshaped)
            x = tf.reshape(x_preprocessed, x_shape)
        return self.encoder(x)

    def decode(self, z):
        z = tf.identity(z)  # in case z is not a Tensor already...
        num_dim = len(z.shape)
        assert num_dim > 2
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        return self.decoder(tf.transpose(z, perm=perm))

    def __call__(self, inputs):
        return self.decode(self.encode(inputs).sample()).sample()

    def generate(self, noise=None, num_samples=1):
        if noise is None:
            noise = tf.random_normal(shape=(num_samples, self.latent_dim))
        return self.decode(noise)
    
    def _get_prior(self):
        if self.prior is None:
            self.prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros([self.latent_dim, self.time_length], dtype=tf.float32),
                scale_diag=tf.ones([self.latent_dim, self.time_length], dtype=tf.float32))
        return self.prior

    def compute_nll(self, x, y=None, m_mask=None):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        y = x if y is None else y

        z_sample = self.encode(x).sample()
        x_hat_dist = self.decode(z_sample)
        nll = -x_hat_dist.log_prob(y)  # shape=(BS, TL, D)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            nll = tf.where(m_mask, nll, tf.zeros_like(nll))  # !!! inverse mask, set zeros for observed
        return tf.reduce_sum(nll)

    def compute_mse(self, x, y=None, m_mask=None, binary=False):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        y = x if y is None else y

        z_mean = self.encode(x).mean()           # through mean
        x_hat_mean = self.decode(z_mean).mean()  # shape=(BS, TL, D)
        if binary:
            x_hat_mean = tf.round(x_hat_mean)    # 转化为最近的整数
        mse = tf.math.squared_difference(x_hat_mean, y)
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            mse = tf.where(m_mask, mse, tf.zeros_like(mse))  # !!! inverse mask, set zeros for observed
        return tf.reduce_sum(mse)

    def _compute_loss(self, x, m_mask=None, return_parts=False):
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        x = tf.tile(x, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)

        if m_mask is not None:
            m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = tf.tile(m_mask, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)
            m_mask = tf.cast(m_mask, tf.bool)

        pz = self._get_prior()  # p(z)
        qz_x = self.encode(x)
        z = qz_x.sample()    # ⚠️这里似乎不需要基于整个MVN采样，实际我们只关心covar的对角线
        px_z = self.decode(z)

        nll = -px_z.log_prob(x)  # shape=(M*K*BS, TL, D)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))  # 基本不会出现infinite的值
        if m_mask is not None:  # mask中为1的位置是未观测到的
            nll = tf.where(m_mask, tf.zeros_like(nll), nll)  # if not HI-VAE, m_mask is always zeros
        nll = tf.reduce_sum(nll, [1, 2])  # shape=(M*K*BS)

        if self.K > 1:
            kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            weights = -nll - kl  # shape=(M*K*BS)
            weights = tf.reshape(weights, [self.M, self.K, -1])  # shape=(M, K, BS)

            elbo = reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
            elbo = tf.reduce_mean(elbo)  # scalar
        else:  # if K==1, compute KL analytically
            kl = self.kl_divergence(qz_x, pz)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            elbo = -nll - self.beta * kl  # shape=(M*K*BS) K=1
            elbo = tf.reduce_mean(elbo)  # scalar, averaged over video dim

        if return_parts:
            nll = tf.reduce_mean(nll)  # scalar
            kl = tf.reduce_mean(kl)  # scalar
            return -elbo, nll, kl
        else:
            return -elbo

    def compute_loss(self, x, m_mask=None, return_parts=False):
        del m_mask
        return self._compute_loss(x, return_parts=return_parts)  # VAE的elbo没有mask，而HI-VAE是有的

    def kl_divergence(self, a, b):
        return tfd.kl_divergence(a, b)

    def get_trainable_vars(self):  # 用于model初始化
        self.compute_loss(tf.random.normal(shape=(1, self.time_length, self.data_dim), dtype=tf.float32),
                          tf.zeros(shape=(1, self.time_length, self.data_dim), dtype=tf.float32))
        return self.trainable_variables


class HI_VAE(VAE):
    """ HI-VAE model, where the reconstruction term in ELBO is summed only over observed components """
    def compute_loss(self, x, m_mask=None, return_parts=False):
        return self._compute_loss(x, m_mask=m_mask, return_parts=return_parts)


class GP_VAE(HI_VAE):
    def __init__(self, *args, kernel="cauchy", sigma=1., length_scale=1.0, kernel_scales=1, **kwargs):
        """ Proposed GP-VAE model with Gaussian Process prior
            :param kernel: Gaussial Process kernel ["cauchy", "diffusion", "rbf", "matern"]
            :param sigma: scale parameter for a kernel function
            :param length_scale: length scale parameter for a kernel function
            :param kernel_scales: number of different length scales over latent space dimensions
        """
        super(GP_VAE, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        if isinstance(self.encoder, JointEncoder):
            self.encoder.transpose = True

        # Precomputed KL components for efficiency
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None
        self.prior = None

    def decode(self, z):  # override
        num_dim = len(z.shape)
        assert num_dim > 2
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        return self.decoder(tf.transpose(z, perm=perm))

    def _get_prior(self):
        if self.prior is None:
            # Compute kernel matrices for each latent dimension
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dim - total
                else:
                    multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
            kernel_matrix_tiled = np.concatenate(tiled_matrices)   # [35,10,10]
            assert len(kernel_matrix_tiled) == self.latent_dim

            self.prior = tfd.MultivariateNormalFullCovariance(   # 实际上是一个block diag MOGP
                loc=tf.zeros([self.latent_dim, self.time_length], dtype=tf.float32),
                covariance_matrix=kernel_matrix_tiled)
        return self.prior

    def kl_divergence(self, a, b):
        """ Batched KL divergence `KL(a || b)` for multivariate Normals.
            See https://github.com/tensorflow/probability/blob/master/tensorflow_probability
                       /python/distributions/mvn_linear_operator.py
            It's used instead of default KL class in order to exploit precomputed components for efficiency
        """

        def squared_frobenius_norm(x):
            """Helper to make KL calculation slightly more readable."""
            return tf.reduce_sum(tf.square(x), axis=[-2, -1])

        def is_diagonal(x):
            """Helper to identify if `LinearOperator` has only a diagonal component."""
            return (isinstance(x, tf.linalg.LinearOperatorIdentity) or
                    isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
                    isinstance(x, tf.linalg.LinearOperatorDiag))

        if is_diagonal(a.scale) and is_diagonal(b.scale):
            # Using `stddev` because it handles expansion of Identity cases.
            b_inv_a = (a.stddev() / b.stddev())[..., tf.newaxis]
        else:
            if self.pz_scale_inv is None:
                self.pz_scale_inv = tf.linalg.inv(b.scale.to_dense())
                self.pz_scale_inv = tf.where(tf.math.is_finite(self.pz_scale_inv),
                                             self.pz_scale_inv, tf.zeros_like(self.pz_scale_inv))

            if self.pz_scale_log_abs_determinant is None:
                self.pz_scale_log_abs_determinant = b.scale.log_abs_determinant()

            a_shape = a.scale.shape
            if len(b.scale.shape) == 3:
                _b_scale_inv = tf.tile(self.pz_scale_inv[tf.newaxis], [a_shape[0]] + [1] * (len(a_shape) - 1))
            else:
                _b_scale_inv = tf.tile(self.pz_scale_inv, [a_shape[0]] + [1] * (len(a_shape) - 1))

            b_inv_a = _b_scale_inv @ a.scale.to_dense()

        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            kl_div = (
                    self.pz_scale_log_abs_determinant - a.scale.log_abs_determinant() +  # 这里的scale类似covar的L
                    0.5 * (
                            -tf.cast(a.scale.domain_dimension_tensor(), a.dtype)
                            + squared_frobenius_norm(b_inv_a)
                            + squared_frobenius_norm(b.scale.solve((b.mean() - a.mean())[..., tf.newaxis]))
                    )
            )
        return kl_div

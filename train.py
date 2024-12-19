import sys
import os
import time
from datetime import datetime
from matplotlib import pyplot as plt
import warnings
from absl import app
from absl import flags
import pickle

from lib.models import *
from lib.building_blocks import *
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append("..")


FLAGS = flags.FLAGS

# HMNIST config  # ⚠️被注释掉的实验setting
flags.DEFINE_integer('latent_dim', 256, 'Dimensionality of the latent space')
flags.DEFINE_list('encoder_sizes', [256, 256], 'Layer sizes of the encoder')
flags.DEFINE_list('decoder_sizes', [256, 256, 256], 'Layer sizes of the decoder')
flags.DEFINE_integer('window_size', 3, 'Window size for the inference CNN: Ignored if model_type is not gp-vae')
flags.DEFINE_float('sigma', 1.0, 'Sigma value for the GP prior: Ignored if model_type is not gp-vae')
flags.DEFINE_float('length_scale', 2.0, 'Length scale value for the GP prior: Ignored if model_type is not gp-vae')
flags.DEFINE_float('beta', 0.8, 'Factor to weigh the KL term (similar to beta-VAE)')
flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs')

# Flags with common default values for all three datasets
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('gradient_clip', 1e4, 'Maximum global gradient norm for the gradient clipping during training')
flags.DEFINE_integer('print_epochs', 1, 'print epochs during training')
flags.DEFINE_string('dataset', 'hmnist_random', 'Type of data to be trained on')
flags.DEFINE_integer('seed', 1337, 'Seed for the random number generator')
flags.DEFINE_enum('model_type', 'vae', ['vae', 'hi-vae', 'gp-vae'], 'Type of model to be trained')

flags.DEFINE_integer('cnn_kernel_size', 3, 'Kernel size for the CNN preprocessor')
flags.DEFINE_list('cnn_sizes', [256], 'Number of filters for the layers of the CNN preprocessor')
flags.DEFINE_boolean('save', False, 'save the results')

flags.DEFINE_boolean('banded_covar', False, 'Use a banded covariance matrix instead of a diagonal one for ' +
                                           'the output of the inference network: Ignored if model_type is not gp-vae')
flags.DEFINE_boolean('joint_encoder', False, 'Use a Conv1D Encoder to model q(z) with a diagonal covariance matrix')

flags.DEFINE_integer('batch_size', 64, 'Batch size for training')

flags.DEFINE_integer('M', 1, 'Number of samples for ELBO estimation')
flags.DEFINE_integer('K', 1, 'Number of importance sampling weights')

flags.DEFINE_enum('kernel', 'cauchy', ['rbf', 'diffusion', 'matern', 'cauchy'], 'Kernel to be used for the GP prior: Ignored if model_type is not (m)gp-vae')
flags.DEFINE_integer('kernel_scales', 1, 'Number of different length scales sigma for the GP prior: Ignored if model_type is not gp-vae')


def main(argv):
    del argv  # unused
    np.random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)

    FLAGS.encoder_sizes = [int(size) for size in FLAGS.encoder_sizes]  # [256, 256]
    FLAGS.decoder_sizes = [int(size) for size in FLAGS.decoder_sizes]  # [256, 256, 256]

    if 0 in FLAGS.encoder_sizes:
        FLAGS.encoder_sizes.remove(0)
    if 0 in FLAGS.decoder_sizes:
        FLAGS.decoder_sizes.remove(0)

    #############
    # Load data #
    #############

    data_dir = f"data/hmnist/{FLAGS.dataset}.npz"
    data_dim, time_length, img_shape, num_classes = 784, 10, (28, 28, 1), 10
    decoder = BernoulliDecoder
    data = np.load(data_dir)
    # [50000, 10, 28*28=784]
    x_train_miss = data['x_train_miss'][:50000] if FLAGS.dataset[:7] == 'hmnist' else data['x_train_miss']
    m_train_miss = data['m_train_miss'][:50000] if FLAGS.dataset[:7] == 'hmnist' else data['m_train_miss']

    x_val_full = data['x_test_full']
    x_val_miss = data['x_test_miss']
    m_val_miss = data['m_test_miss']
    # y_val = data['y_test']

    tf_x_train_miss = tf.data.Dataset.from_tensor_slices((x_train_miss, m_train_miss))\
                                     .shuffle(len(x_train_miss)).batch(FLAGS.batch_size).repeat()
    tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss)).batch(FLAGS.batch_size).repeat()
    # tf_x_val_miss = tf.compat.v1.data.make_one_shot_iterator(tf_x_val_miss)

    # Build Conv2D preprocessor for image data
    print("Using CNN preprocessor in the encoder!\n")
    image_preprocessor = ImagePreprocessor(img_shape, FLAGS.cnn_sizes, FLAGS.cnn_kernel_size)

    ###############
    # Build model #
    ###############

    if FLAGS.model_type == "vae":
        model = VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                    encoder_sizes=FLAGS.encoder_sizes, encoder=DiagonalEncoder,  # ⚠️注意这里VAE和HIVAE均用的是mean-field encoder，并且不使用Conv1D，而是使用普通的Dense layers
                    decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                    image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                    beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K)
    elif FLAGS.model_type == "hi-vae":
        model = HI_VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                       encoder_sizes=FLAGS.encoder_sizes, encoder=DiagonalEncoder,
                       decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                       image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                       beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K)
    elif FLAGS.model_type == "gp-vae":
        if FLAGS.banded_covar:
            encoder = BandedJointEncoder
        elif FLAGS.joint_encoder:
            encoder = JointEncoder
        else:
            encoder = DiagonalEncoder
        print(f"Using {encoder.__class__.__name__} as encoder!")
        model = GP_VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                       encoder_sizes=FLAGS.encoder_sizes, encoder=encoder,
                       decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                       kernel=FLAGS.kernel, sigma=FLAGS.sigma,
                       length_scale=FLAGS.length_scale, kernel_scales=FLAGS.kernel_scales,
                       image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                       beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K)
    else:
        raise ValueError("Model type must be one of ['vae', 'hi-vae', 'gp-vae']")

    ########################
    # Training preparation #
    ########################

    print(f"dataset: {FLAGS.dataset}, seed: {FLAGS.seed}, model: {FLAGS.model_type} "
          f"with {model.encoder.__class__.__name__}, lr:{FLAGS.learning_rate}, beta:{FLAGS.beta}, epochs:{FLAGS.num_epochs}")
    print("GPU support: ", tf.test.is_gpu_available(), "Start Training...")
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()   # 这步同时做了GP prior的初始化。
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    print("\nEncoder: ")
    model.encoder.net.summary()
    print("\nDecoder: ")
    model.decoder.net.summary()
    if model.preprocessor is not None:
        print("\nPreprocessor: ")
        model.preprocessor.net.summary()
        print(" ")
    # summary_writer = tf.contrib.summary.create_file_writer(outdir, flush_millis=10000)

    num_steps = FLAGS.num_epochs * len(x_train_miss) // FLAGS.batch_size
    print_interval = num_steps // FLAGS.num_epochs

    ############
    # Training #
    ############

    losses_train = []
    losses_val = []

    s = time.time()
    t0 = time.time()
    # with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    # with tf.contrib.summary.always_record_summaries():
    for i, (x_seq, m_seq) in enumerate(tf_x_train_miss.take(num_steps)):  # [64, 10, 784]
        with tf.GradientTape() as tape:
            tape.watch(trainable_vars)    # ⚠️ 这里面似乎没有GP kernel相关的参数
            loss = model.compute_loss(x_seq, m_mask=m_seq)
            losses_train.append(loss.numpy())
        grads = tape.gradient(loss, trainable_vars)
        grads = [np.nan_to_num(grad) for grad in grads]
        grads, global_norm = tf.clip_by_global_norm(grads, FLAGS.gradient_clip)
        if global_norm >= FLAGS.gradient_clip:
            print(f"!!!!! grad_norm: {global_norm}, gradient clipped at step {i} !!!!!!")
        optimizer.apply_gradients(
            zip(grads, trainable_vars), global_step=tf.compat.v1.train.get_or_create_global_step())

        # Print intermediate results
        if i % (FLAGS.print_epochs * print_interval) == 0:
            print("================================================")
            print(f"Epoch {i // print_interval}, Step {i} Time = {time.time() - t0:2f}")
            print(f"Learning rate: {optimizer._lr} | Global gradient norm: {global_norm:.2f}")
            loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
            print(f"Train loss = {loss:.3f} | NLL = {nll:.3f} | KL = {kl:.3f}")
            t0 = time.time()
    e = time.time()
    print(f"\nTotal training time: {e - s:4f}.\n")

    ##############
    # Evaluation #
    ##############

    print("Evaluation...")
    # Make up full exp name
    timestamp = datetime.now().strftime("%m%d_%H_%M_%S")
    full_exp_name = f"exp_{FLAGS.model_type}_{FLAGS.dataset}/{timestamp}"
    os.makedirs(full_exp_name, exist_ok=True)
    # Split data on batches
    x_val_miss_batches = np.array_split(x_val_miss, FLAGS.batch_size, axis=0)
    x_val_full_batches = np.array_split(x_val_full, FLAGS.batch_size, axis=0)
    m_val_batches = np.array_split(m_val_miss, FLAGS.batch_size, axis=0)
    get_val_batches = lambda: zip(x_val_miss_batches, x_val_full_batches, m_val_batches)

    # Compute NLL and MSE on missing values
    n_missings = m_val_miss.sum()
    nll_miss = np.sum(
        [model.compute_nll(x, y=y, m_mask=m, num_samples=20).numpy() for x, y, m in get_val_batches()]) / n_missings
    mse_miss = np.sum(
        [model.compute_mse(x, y=y, m_mask=m, binary=True).numpy() for x, y, m in get_val_batches()]) / n_missings
    mse_miss_not_round = np.sum(
        [model.compute_mse(x, y=y, m_mask=None, binary=False).numpy() for x, y, m in get_val_batches()]
    ) / x_val_full.size
    print(f"NLL miss: {nll_miss},"
          f"\tMSE rounded at missing pixels: {mse_miss}, "
          f"\tMSE not rounded at all pixels: {mse_miss_not_round}")

    # Save imputed values
    if FLAGS.save:
        z_mean = [model.encode(x_batch).mean().numpy() for x_batch in x_val_miss_batches]
        x_val_imputed = np.vstack([model.decode(z_batch).mean().numpy() for z_batch in z_mean])

        # impute gt observed values
        x_val_imputed_copy = np.copy(x_val_imputed)
        x_val_imputed_copy[m_val_miss == 0] = x_val_miss[m_val_miss == 0]
        np.save(os.path.join(full_exp_name, "imputed_fill_observed_one"), x_val_imputed)
        everything_for_imgs = {
            'y_test_full': x_val_full, 'y_test_miss': x_val_miss, 'y_rec': x_val_imputed,
            'NLL': nll_miss, 'MSE': mse_miss, 'MSE_non_round': mse_miss_not_round
        }
        imgs_pickle_path = os.path.join(full_exp_name, 'everything_for_imgs.pkl')
        with open(imgs_pickle_path, 'wb') as f:
            pickle.dump(everything_for_imgs, f)

    # Visualize reconstructions
    img_index, num_imgs = 0, 10
    fig, axes = plt.subplots(nrows=3, ncols=num_imgs, figsize=(2*num_imgs, 6))

    x_hat = model.decode(model.encode(x_val_miss[img_index: img_index+1]).mean()).mean().numpy()

    for i in range(num_imgs):
        axes[0, i].imshow(x_val_miss[img_index, i].reshape(28, 28), cmap='gray')
        axes[1, i].imshow(x_hat[0, i].reshape(28, 28), cmap='gray')
        axes[2, i].imshow(x_val_full[img_index, i].reshape(28, 28), cmap='gray')
        for j in range(2):
            axes[j, i].axis('off')

    suptitle = FLAGS.model_type + f" reconstruction, MSE missing = {mse_miss}"
    fig.suptitle(suptitle, size=18)
    fig.savefig(os.path.join(full_exp_name, f"{FLAGS.dataset}_{model.encoder.__class__.__name__}.pdf"))

    print("Evaluation finished. Results are saved at:")
    print(f"{full_exp_name}")


if __name__ == '__main__':
    app.run(main)

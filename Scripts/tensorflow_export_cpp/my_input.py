__author__ = 'HANEL'

import tensorflow as tf
import os
import glob
import numpy as np
import csv
import my_cifar

Data_PATH = '../mcifar_data/'

# Parameters
num_classes = 10
IMAGE_SIZE = 32
IMAGE_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, 3]
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 50000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_string('train_dir', '../my_data_raw', 'Directory with the training ckpt.')

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


def _dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  print(labels_one_hot[0])
  return labels_one_hot


def _label_to_int(labels):
  categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  new_labels = []

  for label in labels:
    new_labels.append(categories.index(label[1]))
  return new_labels


'''Read Images and Labels normally, with python :))) :D'''


def read_labels_from(path=Data_PATH, num_classes=num_classes, one_hot=False):
  print('Reading labels')
  with open(os.path.join(path, 'trainLabels.csv'), 'r') as dest_f:
    data_iter = csv.reader(dest_f)
    train_labels = [data for data in data_iter]

  # pre process labels to int
  train_labels = _label_to_int(train_labels)
  train_labels = np.array(train_labels, dtype=np.uint32)

  # if one_hot:
  #   labels_one_hot = _dense_to_one_hot(train_labels, num_classes)
  #   labels_one_hot = np.asarray(labels_one_hot)
  #   return labels_one_hot

  return train_labels


def read_images_from(path=Data_PATH):
  images = []
  png_files_path = glob.glob(os.path.join(path, 'train/', '*.[pP][nN][gG]'))
  for filename in png_files_path:
    im = Image.open(filename)  # .convert("L")  # Convert to greyscale
    im = np.asarray(im, np.uint8)
    # print(type(im))
    # get only images name, not path
    image_name = filename.split('/')[-1].split('.')[0]
    images.append([int(image_name), im])

  images = sorted(images, key=lambda image: image[0])

  images_only = [np.asarray(image[1], np.uint8) for image in images]  # Use unint8 or you will be !!!
  images_only = np.array(images_only)

  print(images_only.shape)
  return images_only


''' Decode TFRecords '''


def read_and_decode1(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    dense_keys=['image_raw', 'label'],
    # Defaults are not specified since both keys are required.
    dense_types=[tf.string, tf.int64])

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)

  # image.set_shape([32, 32, 3])
  # print('len of readed image is ', image)
  # Image._show(Image.fromarray(np.reshape(image, [32, 32, 3])))

  image = tf.reshape(image, [my_cifar.n_input])
  image.set_shape([my_cifar.n_input])
  # image.set_shape([32*32*3])
  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32)
  # image = tf.cast(image, tf.float32) * (1. / 255)
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


# Decode TFRecords
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example, features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64)
    }
  )

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)

  image = tf.reshape(image, [my_cifar.n_input])
  image.set_shape([my_cifar.n_input])

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs):
  """Reads input ckpt num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) ckpt.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input ckpt, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs, name='string_input_producer')

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)
    print('1- image shape is ', image.get_shape())

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    images, sparse_labels = tf.train.shuffle_batch(
      [image, label], batch_size=batch_size, num_threads=5,
      capacity=min_queue_examples + 3 * batch_size, enqueue_many=False,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=min_queue_examples, name='batching_shuffling')
    print('1.1- label batch shape is ', sparse_labels.get_shape())

    return images, sparse_labels


def inputs_val(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs):
  """Reads input ckpt num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) ckpt.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input ckpt, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs:
    num_epochs = None

  filename = os.path.join(FLAGS.train_dir,
                          VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      [filename])

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)
    print('1- image shape is ', image.get_shape())

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
      [image, label], batch_size=batch_size, num_threads=3,
      capacity=300 + 3 * batch_size, enqueue_many=False,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=300)
    print('1.1- images batch shape is ', images.get_shape())

    return images, sparse_labels


def distorted_inputs(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs):
  """Construct distorted input for CIFAR training using the Reader ops.

  Raises:
    ValueError: if no data_dir

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not num_epochs:
    num_epochs = None
  filename = os.path.join(FLAGS.train_dir,
                          TRAIN_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

  # Reshape to [32, 32, 3] as distortion methods need this shape
  image = tf.reshape(image, IMAGE_SHAPE)
  image.set_shape(IMAGE_SHAPE)

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(image)

  # Because these operations are not commutative, consider randomizing
  # randomize the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Reshape back to original placeholder shape and other architecture
  image = tf.reshape(float_image, [my_cifar.n_input])
  image.set_shape([my_cifar.n_input])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                 batch_size=batch_size,
                                                 num_threads=5,
                                                 capacity=min_queue_examples + 3 * batch_size,
                                                 enqueue_many=False,
                                                 # Ensures a minimum amount of shuffling of examples.
                                                 min_after_dequeue=min_queue_examples,
                                                 name='batching_shuffling_distortion')

  return images, sparse_labels


def main(argv=None):
  return 0


if __name__ == '__main__':
  tf.app.run()

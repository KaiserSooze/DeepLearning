__author__ = 'HANEL'

from datetime import datetime
import os.path
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import my_cifar
import my_input
import freeze_graph

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', 'tmp/my-model',
                           """Directory where to write model proto """
                           """ to import in c++""")
tf.app.flags.DEFINE_string('train_dirr', 'tmp/log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_string('eval_dir', 'tmp/log_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/ckpt',
                           """Directory where to read model checkpoints.""")

# Parameters
display_step = 1
IMAGE_PIXELS = 32 * 32 * 3
NEW_LINE = '\n'

checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded ckpt in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test ckpt sets.
  # batch_size = -1
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

  return images_placeholder, labels_placeholder


def calculate_accuracy(logit, labels, feed):
  # Calculate accuracy
  acc, n_correct = my_cifar.evaluation(logit, labels)

  return acc, n_correct


def train(TRAIN=True):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():

    global_step = tf.Variable(0, trainable=False)

    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    # Get images and labels for CIFAR-10.
    images, labels = my_input.distorted_inputs()
    val_images, val_labels = my_input.inputs(False)
    print('2- images shape is ', images.get_shape())
    print('3- labels shape is ', labels.get_shape())

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = my_cifar.inference(images_placeholder)
    print('4- logits shape is ', logits.get_shape())

    # Calculate loss.
    loss = my_cifar.loss(logits, labels_placeholder)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = my_cifar.training(loss, global_step)

    # Calculate accuracy ##
    acc, n_correct = my_cifar.evaluation(logits, labels_placeholder)

    # Create a saver.
    saver = tf.train.Saver()

    tf.scalar_summary('Acc', acc)
    tf.scalar_summary('Loss', loss)
    tf.image_summary('Images', tf.reshape(images, shape=[-1, 32, 32, 3]), max_images=10)
    tf.image_summary('Val Images', tf.reshape(val_images, shape=[-1, 32, 32, 3]), max_images=10)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dirr,
                                            graph_def=sess.graph_def)

    if TRAIN:
      try:
        while not coord.should_stop():

          for step in xrange(FLAGS.max_steps):

            images_r, labels_r = sess.run([images, labels])
            images_val_r, labels_val_r = sess.run([val_images, val_labels])

            train_feed = {images_placeholder: images_r,
                          labels_placeholder: labels_r}

            val_feed = {images_placeholder: images_val_r,
                        labels_placeholder: labels_val_r}

            start_time = time.time()

            _, loss_value = sess.run([train_op, loss], feed_dict=train_feed)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % display_step == 0:
              num_examples_per_step = FLAGS.batch_size
              examples_per_sec = num_examples_per_step / duration
              sec_per_batch = float(duration)

              format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print_str_loss = format_str % (datetime.now(), step, loss_value,
                                             examples_per_sec, sec_per_batch)
              print (print_str_loss)
              summary_str = sess.run([summary_op], feed_dict=train_feed)
              summary_writer.add_summary(summary_str[0], step)

            if step % (display_step * 5) == 0:
              acc_value, num_corroect = sess.run([acc, n_correct], feed_dict=train_feed)

              format_str = '%s: step %d,  train acc = %.2f, n_correct= %d'
              print_str_train = format_str % (datetime.now(), step, acc_value, num_corroect)
              print (print_str_train)

            # Save the model checkpoint periodically.
            if (step + 1) % (display_step * 10) == 0 or (step + 1) == FLAGS.max_steps:
              val_acc_r, val_n_correct_r = sess.run([acc, n_correct], feed_dict=val_feed)  # , feed_dict=val_feed

              frmt_str = 'Step %d, Val Acc = %.2f, num correct = %d'
              print_str_val = frmt_str % (step, val_acc_r, val_n_correct_r)
              print(print_str_val)

              # checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
              # saver.save(sess, checkpoint_path, global_step=step, latest_filename=checkpoint_state_name)

              checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "saved_checkpoint")
              saver.save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)

      except tf.errors.OutOfRangeError:
        print ('Done training -- epoch limit reached')

      finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        '''
         TODO #3.1: Start freezing the graph when training finished
        '''
        freeze_my_graph(sess)

      # Wait for threads to finish.
      coord.join(threads)
      sess.close()

    # If you define TRAIN argument to False, so it will load from the checkpoint file and freeze.
    else:
      '''
         TODO #3.2: You can also freeze the graph from the latest checkpoint if you don't want to wait for a long time.
      '''
      freeze_my_graph(sess)

def freeze_my_graph(sess):

  tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.model_dir, input_graph_name)

  # We save out the graph to disk, and then call the const conversion
  # routine.

  checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "saved_checkpoint")
  input_graph_path = os.path.join(FLAGS.model_dir, input_graph_name)
  input_saver_def_path = ""
  input_binary = False
  input_checkpoint_path = checkpoint_prefix + "-0"
  # input_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt') + "-0"
  # input_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-299')
  output_node_names = "Dense2/output_node"
  restore_op_name = "save/restore_all"
  filename_tensor_name = "save/Const:0"
  output_graph_path = os.path.join(FLAGS.model_dir, output_graph_name)
  clear_devices = False

  freeze_graph.freeze_graph(input_graph_path,
                            input_saver_def_path,
                            input_binary,
                            input_checkpoint_path,
                            output_node_names,
                            restore_op_name,
                            filename_tensor_name,
                            output_graph_path,
                            clear_devices)


def main(argv=None):
  '''
  Todo: If you put TRAIN argument True(default value) then the model will train and freeze it when the training
  is finished, if you put it False, then it will load the latest checkpoint and freeze that model.
  '''
  train(TRAIN=False)


if __name__ == '__main__':
  tf.app.run()

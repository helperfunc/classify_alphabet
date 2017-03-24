import tensorflow as tf
from model.lenet5_tflayers import Model
from misc.datasets import MnistDataset
import math
import numpy as np
from datetime import datetime

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/Users/huixu/Documents/codelabs/alphabet2cla/logs_test/', 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('eval_dir', '/Users/huixu/Documents/codelabs/alphabet2cla/logs_eval/',
                           'Where to save the logs for visualization in TensorBoard.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 52832,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 676

LEARNING_RATE = 1E-4
def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / BATCH_SIZE))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f, true num/total num: %d / %d' % (datetime.now(), precision, true_count, total_sample_count))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        #eval_data = FLAGS.eval_data == 'test'
        #images, labels = cifar10.inputs(eval_data=eval_data)
        dataset = MnistDataset(BATCH_SIZE)
        images, labels = dataset.get_eval_batch_images()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # Set model params
        model_params = {"learning_rate": LEARNING_RATE}
        # Training data, nodes in a graph
        #is_train = tf.placeholder(tf.bool, name="is_train")
        model = Model(images, labels, False, model_params)
        logits = model.inference

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        #variable_averages = tf.train.ExponentialMovingAverage(
        #    cifar10.MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    #cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()

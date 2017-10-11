
import tensorflow as tf
import numpy as np
from PIL import Image


class DataSetProvider(object):

    _data_pool = None

    @property
    def data_pool(self):
        if not self._data_pool:
            from tensorflow.examples.tutorials.mnist import input_data
            self._data_pool = input_data.read_data_sets("./mnist/data/", one_hot=True)
        return self._data_pool

    def iterate_batch(self):
        num_examples = self.data_pool.train.num_examples
        cursor = 0
        step = 100

        while cursor < num_examples - step:
            cursor += step
            yield self.data_pool.train.next_batch(step)

    def get_testdata(self):
        return (self.data_pool.test.images, self.data_pool.test.labels)


class AbstractAgent(object):
    MODEL_PATH = "./model"
    MODEL_CKP_PATH = "./model/agent.ckpt"
    LOG_PATH = "./log/"
    LEARNING_RATE = 0.001

    global_step = None
    session = None
    saver = None

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self._build_model()

    def _build_model(self):
        """ override this """
        pass

    @property
    def run(self):
        """ shortcut to session.run """
        return self.session.run

    def __enter__(self):
        self.session = tf.Session()
        self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(AbstractAgent.MODEL_PATH)

        if ckpt and tf.train.checkpoint_exists(AbstractAgent.MODEL_PATH):
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            self.session.run(tf.global_variables_initializer())

        return self

    def __exit__(self, type, value, traceback):
        print("Saving session..")
        if self.saver:
            self.saver.save(
                self.session,
                AbstractAgent.MODEL_CKP_PATH,
                global_step=self.global_step
            )

        if self.session:
            self.session.close()

        print("Session closed")

    def _train(self, dataset_provider):
        """ override this """
        pass

    def _test(self, dataset_provider):
        """ override this """
        pass

    def _predict(self, x):
        """ override this """
        pass

    def train(self, dataset_provider):
        self._train(dataset_provider)

    def test(self, dataset_provider):
        """ prints accuracy """
        self._test(dataset_provider)

    def predict(self, x):
        self._predict(x)


class CNNAgent(AbstractAgent):
    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")
        self.Y = tf.placeholder(tf.float32, [None, 10], name="Y")
        self.is_training = tf.placeholder(tf.bool)

        L1, L2, L3 = None, None, None
        with tf.name_scope("layer1"):
            L1 = tf.layers.dropout(
                tf.layers.max_pooling2d(
                    tf.layers.conv2d(self.X, 32, [3, 3]),
                    [2, 2], [2, 2]),
                0.7, self.is_training)

        with tf.name_scope("layer2"):
            L2 = tf.layers.dropout(
                tf.layers.max_pooling2d(
                    tf.layers.conv2d(L1, 64, [3, 3]),
                    [2, 2], [2, 2]),
                0.7, self.is_training)

        with tf.name_scope("layer3"):
            L3 = tf.layers.dropout(
                tf.layers.dense(
                    tf.contrib.layers.flatten(L2),
                    256, activation=tf.nn.relu),
                0.5, self.is_training)

        self.model = tf.layers.dense(L3, 10, activation=None)
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(AbstractAgent.LEARNING_RATE)\
            .minimize(self.cost, global_step=self.global_step)

    def _train(self, dataset_provider):
        for xs, ys in dataset_provider.iterate_batch():
            feed_dict = {
                self.X: xs.reshape(-1, 28, 28, 1),
                self.Y: ys,
                self.is_training: True
            }
            _, cost = self.run(
                [self.optimizer, self.cost],
                feed_dict=feed_dict)

            global_step_value = self.run(self.global_step)
            print("Global step: ", global_step_value)

    def _test(self, dataset_provider):
        is_correct = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        xs, ys = dataset_provider.get_testdata()
        feed_dict = {
            self.X: xs.reshape(-1, 28, 28, 1),
            self.Y: ys
        }
        acc_value = self.run(accuracy, feed_dict=feed_dict)
        print("Accuracy: ", acc_value)

    def _predict(self, x):
        res = self.run(tf.argmax(self.model, 1), feed_dict={self.X: x})
        print(res)


img = Image.open("target_image.png")
target_image_data = (1. - (np.asarray(img) / 255)).reshape(-1, 28, 28, 1)

# used to train and test CNN agent
# dataset_provider = DataSetProvider()
with CNNAgent() as agent:
    # agent.train(dataset_provider)
    # agent.test(dataset_provider)
    agent.predict(target_image_data)


import os

import tensorflow as tf
import numpy as np
from PIL import Image


class DataSetProvider(object):

    _data_pool = None

    def __init__(self):
        print("initializing data set provider..")

    @property
    def data_pool(self):
        if not self._data_pool:
            w = 30
            h = 30
            self._data_pool = {}
            img = Image.open("train_data/alphabet.png").convert("L")
            y_count = (int)(img.size[1] / h)
            print(y_count)
            for x in range(26):
                dataset = []
                for y in range(y_count):
                    rect = (x * w, y * h, x * w + w, y * h + h)
                    data = (1. - (np.asarray(img.crop(rect)) / 255.)).reshape(-1, w, h, 1)
                    dataset.append(data)

                self._data_pool[x] = np.array(dataset)
        return self._data_pool

    def iterate_batch(self):
        REPEAT_TIME = 1
        for _ in range(REPEAT_TIME):
            for x, dataset in self.data_pool.items():
                label = np.zeros(shape=(26))
                label[x] = 1

                for data in dataset:
                    yield data, label


class AbstractAgent(object):
    MODEL_PATH = "./model"
    MODEL_CKP_PATH = "./model/agent.ckpt"
    LOG_PATH = "./log/"
    LEARNING_RATE = 0.001

    global_step = None
    session = None
    saver = None
    writer = None

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        print("initializing model..")
        self._build_model()

        print("initialized model!")

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
            print("loading saved model checkpoint")
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print("no checkpoint is found,")
            self.session.run(tf.global_variables_initializer())

        tf.summary.histogram("model", self.model)
        tf.summary.scalar('cost', self.cost)
        self.writer = tf.summary.FileWriter("./logs", self.session.graph)
        return self

    def __exit__(self, type, value, traceback):
        print("disposing agent, saving session..")

        if self.saver:
            self.saver.save(
                self.session,
                AbstractAgent.MODEL_CKP_PATH,
                global_step=self.global_step
            )

        if self.session:
            self.session.close()

        print("session disposed!")

    def _train(self, dataset_provider):
        """ override this """
        pass

    def _predict(self, x):
        """ override this """
        pass

    def train(self, dataset_provider):
        self._train(dataset_provider)

    def predict(self, x):
        return self._predict(x)


class CNNAgent(AbstractAgent):
    def _build_model(self) -> None:
        self.LEARNING_RATE = 0.01
        self.X = tf.placeholder(tf.float32, [None, 30, 30, 1], name="X")
        self.Y = tf.placeholder(tf.float32, [None, 26], name="Y")
        self.is_training = tf.placeholder(tf.bool)

        L1, L2, L3 = None, None, None
        with tf.name_scope("layer1"):
            L1 = tf.layers.dropout(
                tf.layers.max_pooling2d(
                    tf.layers.conv2d(self.X, filters=32, kernel_size=3),
                    pool_size=2, strides=2),
                0.8, self.is_training)

        with tf.name_scope("layer2"):
            L2 = tf.layers.dropout(
                tf.layers.average_pooling2d(
                    tf.layers.conv2d(L1, filters=64, kernel_size=3),
                    pool_size=2, strides=2),
                0.8, self.is_training)

        with tf.name_scope("layer3"):
            L3 = tf.layers.dropout(
                tf.layers.max_pooling2d(
                    tf.layers.conv2d(L2, filters=64, kernel_size=3),
                    pool_size=2, strides=2),
                0.8, self.is_training)

        with tf.name_scope("layer4"):
            L4 = tf.layers.dropout(
                tf.layers.dense(
                    tf.contrib.layers.flatten(L3),
                    256, activation=tf.nn.relu),
                0.6, self.is_training)

        self.model = tf.layers.dense(L4, 26, activation=None)
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(AbstractAgent.LEARNING_RATE)\
            .minimize(self.cost, global_step=self.global_step)

    def _train(self, dataset_provider):
        merge = tf.summary.merge_all()
        i = 0
        for xs, ys in dataset_provider.iterate_batch():
            x_feed = xs.reshape(-1, 30, 30, 1)
            y_feed = ys.reshape(-1, 26)
            feed_dict = {
                self.X: x_feed,
                self.Y: y_feed,
                self.is_training: True
            }

            _, cost = self.run(
                [self.optimizer, self.cost],
                feed_dict=feed_dict)

            i += 1
            if i % 30 == 0:

                # print step
                global_step_value = self.run(self.global_step)
                print("Global step: ", global_step_value, "Cost: ", cost)

                # report to tensor board
                summary = self.run(merge, feed_dict=feed_dict)
                self.writer.add_summary(summary, global_step=global_step_value)

                # print progress
                print("Y: ", chr(self.run(tf.argmax(self.Y, 1), feed_dict={self.Y: y_feed}) + 97))

    def _predict(self, x):
        return self.run(tf.argmax(self.model, 1), feed_dict={self.X: x})


# entry point
if __name__ == "__main__":
    is_trainmode = False
    dataset_provider = None
    target_data_set = None

    # used to train and test CNN agent
    if is_trainmode:
        dataset_provider = DataSetProvider()
    else:
        target_data_set = {}
        for filename in os.listdir("target_images"):
            img = Image.open("target_images/{}".format(filename)).resize((30, 30)).convert("L")
            target_data_set[filename.split(".")[0]] = \
                (1. - (np.asarray(img) / 255)).reshape(1, 30, 30, 1)

    with CNNAgent() as agent:
        if is_trainmode:
            agent.train(dataset_provider)
        else:
            for key, data in target_data_set.items():
                res = agent.predict(data)
                print("Key: {}, Read: {}".format(key, chr(res + 97)))

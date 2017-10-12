
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
            # for x in range(26):
            for x in range(3):
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

    def _train(self, dataset_provider=None):
        """ override this """
        pass

    def _evaluate(self):
        """ override this """
        pass

    def _predict(self, x):
        """ override this """
        pass

    def train(self, dataset_provider=None):
        self.merge = tf.summary.merge_all()
        self._train(dataset_provider)

    def evaluate(self):
        self._evaluate()

    def predict(self, x):
        return self._predict(x)


class GoogLeNetAgent(AbstractAgent):
    def _build_model(self) -> None:
        """
        Just following the googlenet example:
         - https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py
        """

        import tflearn
        from tflearn.layers import core
        from tflearn.layers import conv
        from tflearn.layers import normalization
        from tflearn.layers import merge_ops
        from tflearn.layers import estimator

        from tflearn.datasets import oxflower17

        w = 227
        h = 227
        self.X, self.Y = oxflower17.load_data(one_hot=True, resize_pics=(w, h))
        print("X, Y shape: ", self.X.shape, self.Y.shape)

        data_input = core.input_data(shape=[None, w, h, 3])
        conv1_7_7 = conv.conv_2d(data_input, 64, 7, strides=2, activation="relu", name="conv1_7_7_s2")
        pool1_3_3 = normalization.local_response_normalization(
            conv.max_pool_2d(conv1_7_7, 3, strides=2))
        conv2_3_3 = normalization.local_response_normalization(
            conv.conv_2d(
                conv.conv_2d(
                    pool1_3_3, 64, 1, activation="relu", name="conv2_3_3"),
                182, 3, activation="relu"))
        pool2_3_3 = conv.max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name="pool2_3_3_s2")

        # 3a
        inception_3a_1_1 = conv.conv_2d(
            pool2_3_3, 64, 1, activation="relu", name="inception_3a_1_1")
        inception_3a_3_3 = conv.conv_2d(
            conv.conv_2d(pool2_3_3, 96, 1, activation="relu"),
            128, filter_size=3, activation="relu", name="inception_3a_3_3")
        inception_3a_5_5 = conv.conv_2d(
            conv.conv_2d(pool2_3_3, 16, filter_size=1, activation="relu"),
            32, filter_size=5, activation="relu", name="inception_3a_5_5")
        inception_3a_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name="inception_3a_pool"),
            32, filter_size=1, activation="relu", name="inception_3a_pool_1_1")
        inception_3a_output = merge_ops.merge(
            [inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
            mode="concat", axis=3)

        # 3b
        inception_3b_1_1 = conv.conv_2d(
            inception_3a_output, 128, 1, activation="relu", name="inception_3b_1_1")
        inception_3b_3_3 = conv.conv_2d(
            conv.conv_2d(inception_3a_output, 128, 1, activation="relu"),
            192, filter_size=3, activation="relu", name="inception_3b_3_3")
        inception_3b_5_5 = conv.conv_2d(
            conv.conv_2d(inception_3a_output, 32, filter_size=1, activation="relu"),
            96, filter_size=5, activation="relu", name="inception_3b_5_5")
        inception_3b_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name="inception_3b_pool"),
            64, filter_size=1, activation="relu", name="inception_3b_pool_1_1")
        inception_3b_output = merge_ops.merge(
            [inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
            mode="concat", axis=3, name="inception_3b_output")

        pool3_3_3 = conv.max_pool_2d(
            inception_3b_output, kernel_size=3, strides=2, name="pool3_3_3")

        # 4a
        inception_4a_1_1 = conv.conv_2d(
            pool3_3_3, 192, filter_size=1, activation="relu", name="inception_4a_1_1")
        inception_4a_3_3 = conv.conv_2d(
            conv.conv_2d(pool3_3_3, 96, filter_size=1, activation="relu"),
            208, filter_size=3, activation="relu", name="inception_4a_3_3")
        inception_4a_5_5 = conv.conv_2d(
            conv.conv_2d(pool3_3_3, 16, filter_size=1, activation="relu"),
            48, filter_size=5, activation="relu", name="inception_4a_5_5")
        inception_4a_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(pool3_3_3, kernel_size=3, strides=1),
            64, filter_size=1, name="inception_4a_pool")
        inception_4a_output = merge_ops.merge(
            [inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
            mode="concat", axis=3, name="inception_4a_output")

        # 4b
        inception_4b_1_1 = conv.conv_2d(
            inception_4a_output, 160, filter_size=1, activation="relu", name="inception_4b_1_1")
        inception_4b_3_3 = conv.conv_2d(
            conv.conv_2d(inception_4a_output, 112, filter_size=1, activation="relu"),
            224, filter_size=3, activation="relu", name="inception_4b_3_3")
        inception_4b_5_5 = conv.conv_2d(
            conv.conv_2d(inception_4a_output, 24, filter_size=1, activation="relu"),
            64, filter_size=5, activation="relu", name="inception_4b_5_5")
        inception_4b_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(inception_4a_output, kernel_size=3, strides=1),
            64, filter_size=1, name="inception_4b_pool")
        inception_4b_output = merge_ops.merge(
            [inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
            mode="concat", axis=3, name="inception_4b_output")

        # 4c
        inception_4c_1_1 = conv.conv_2d(
            inception_4b_output, 128, filter_size=1, activation="relu", name="inception_4c_1_1")
        inception_4c_3_3 = conv.conv_2d(
            conv.conv_2d(inception_4b_output, 128, filter_size=1, activation="relu"),
            256, filter_size=3, activation="relu", name="inception_4c_3_3")
        inception_4c_5_5 = conv.conv_2d(
            conv.conv_2d(inception_4b_output, 24, filter_size=1, activation="relu"),
            64, filter_size=5, activation="relu", name="inception_4c_5_5")
        inception_4c_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(inception_4b_output, kernel_size=3, strides=1),
            64, filter_size=1, name="inception_4c_pool")
        inception_4c_output = merge_ops.merge(
            [inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
            mode="concat", axis=3, name="inception_4c_output")

        # 4d
        inception_4d_1_1 = conv.conv_2d(
            inception_4c_output, 112, filter_size=1, activation="relu", name="inception_4d_1_1")
        inception_4d_3_3 = conv.conv_2d(
            conv.conv_2d(inception_4c_output, 144, filter_size=1, activation="relu"),
            288, filter_size=3, activation="relu", name="inception_4d_3_3")
        inception_4d_5_5 = conv.conv_2d(
            conv.conv_2d(inception_4c_output, 32, filter_size=1, activation="relu"),
            64, filter_size=5, activation="relu", name="inception_4d_5_5")
        inception_4d_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(inception_4c_output, kernel_size=3, strides=1),
            64, filter_size=1, name="inception_4d_pool")
        inception_4d_output = merge_ops.merge(
            [inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
            mode="concat", axis=3, name="inception_4d_output")

        # 4e
        inception_4e_1_1 = conv.conv_2d(
            inception_4d_output, 256, filter_size=1, activation="relu", name="inception_4d_1_1")
        inception_4e_3_3 = conv.conv_2d(
            conv.conv_2d(inception_4d_output, 160, filter_size=1, activation="relu"),
            320, filter_size=3, activation="relu", name="inception_4d_3_3")
        inception_4e_5_5 = conv.conv_2d(
            conv.conv_2d(inception_4d_output, 32, filter_size=1, activation="relu"),
            128, filter_size=5, activation="relu", name="inception_4d_5_5")
        inception_4e_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(inception_4d_output, kernel_size=3, strides=1),
            128, filter_size=1, name="inception_4d_pool")
        inception_4e_output = merge_ops.merge(
            [inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1],
            mode="concat", axis=3, name="inception_4d_output")

        pool4_3_3 = conv.max_pool_2d(
            inception_4e_output, kernel_size=3, strides=2, name="pool4_3_3")

        # 5a
        inception_5a_1_1 = conv.conv_2d(
            pool4_3_3, 256, filter_size=1, activation="relu", name="inception_5a_1_1")
        inception_5a_3_3 = conv.conv_2d(
            conv.conv_2d(pool4_3_3, 160, filter_size=1, activation="relu"),
            320, filter_size=3, activation="relu", name="inception_5a_3_3")
        inception_5a_5_5 = conv.conv_2d(
            conv.conv_2d(pool4_3_3, 32, filter_size=1, activation="relu"),
            128, filter_size=5, activation="relu", name="inception_5a_5_5")
        inception_5a_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(pool4_3_3, kernel_size=3, strides=1),
            128, filter_size=1, name="inception_5a_pool")
        inception_5a_output = merge_ops.merge(
            [inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],
            mode="concat", axis=3, name="inception_5a_output")

        # 5b
        inception_5b_1_1 = conv.conv_2d(
            inception_5a_output, 256, filter_size=1, activation="relu", name="inception_5b_1_1")
        inception_5b_3_3 = conv.conv_2d(
            conv.conv_2d(inception_5a_output, 160, filter_size=1, activation="relu"),
            320, filter_size=3, activation="relu", name="inception_5b_3_3")
        inception_5b_5_5 = conv.conv_2d(
            conv.conv_2d(inception_5a_output, 32, filter_size=1, activation="relu"),
            128, filter_size=5, activation="relu", name="inception_5b_5_5")
        inception_5b_pool_1_1 = conv.conv_2d(
            conv.max_pool_2d(inception_5a_output, kernel_size=3, strides=1),
            128, filter_size=1, name="inception_5b_pool")
        inception_5b_output = merge_ops.merge(
            [inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],
            mode="concat", axis=3, name="inception_5b_output")

        pool5_7_7 = core.dropout(
            conv.avg_pool_2d(inception_5b_output, kernel_size=7, strides=1),
            0.4)

        # fc
        loss = core.fully_connected(pool5_7_7, 17, activation="softmax")
        network = estimator.regression(
            loss,
            optimizer="momentum",
            loss="categorical_crossentropy",
            learning_rate=self.LEARNING_RATE)
        self.model = tflearn.DNN(
            network, checkpoint_path="model_googlenet",
            max_checkpoints=1, tensorboard_verbose=2)

    def _train(self, dataset_provider=None):
        self.model.fit(
            self.X, self.Y,
            n_epoch=1000, validation_set=0.1,
            shuffle=True, show_metric=True,
            batch_size=64, snapshot_step=200,
            snapshot_epoch=False, run_id="googlenet_oxflower17")

    def _predict(self, x):
        return self.run(tf.argmax(self.model, 1), feed_dict={self.X: x})


# entry point
if __name__ == "__main__":
    is_trainmode = True

    # used to train and test GoogLeNet agent
    with GoogLeNetAgent() as agent:
        if is_trainmode:
            agent.train()

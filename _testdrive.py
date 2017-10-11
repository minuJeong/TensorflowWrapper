
import tensorflow as tf
import numpy as np


LEARNING_RATE = 0.1

x_data = np.random.rand(100, 30)
means = []
for x in x_data:
    means.append(np.mean(x))
y_data = np.array(means).reshape(100, 1)
print(y_data)

X = tf.placeholder(dtype=tf.float32, name="X")
Y = tf.placeholder(dtype=tf.float32, name="Y")

W = tf.Variable(0, dtype=tf.float32, name="weight")
b = tf.Variable(0, dtype=tf.float32, name="bias")

with tf.name_scope("layer_1"):
    layer1_weight = tf.Variable(tf.random_uniform([30, 100], -1., 1.), name="layer1_weight")
    layer1 = tf.nn.relu(tf.matmul(X, layer1_weight))

with tf.name_scope("layer_2"):
    layer2_weight = tf.Variable(tf.random_uniform([100, 50], -1., 1.), name="layer2_weight")
    layer2 = tf.nn.relu(tf.matmul(layer1, layer2_weight))

with tf.name_scope("output"):
    layer3_weight = tf.Variable(tf.random_uniform([50, 1], -1., 1.), name="layer3_weight")
    model = tf.matmul(layer2, layer3_weight)

with tf.name_scope("optimizer"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_operation = optimizer.minimize(cost)

    tf.summary.scalar("cost", cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', session.graph)

    for step in range(1000):
        _, cost_value = session.run([train_operation, cost], feed_dict={X: x_data, Y: y_data})
        print(step, cost_value)

        summary = session.run(merged, feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, step)

    prediction = tf.argmax(model, 0)
    target = tf.argmax(Y, 0)

    is_correct = tf.equal(prediction, target)
    accuracy = session.run(
        tf.reduce_mean(tf.cast(is_correct, tf.float32)),
        feed_dict={X: x_data, Y: y_data}
    )

    pres = session.run(prediction, feed_dict={X: x_data})
    tres = session.run(target, feed_dict={Y: y_data})
    print(pres, tres)
    print("accuracy: ", accuracy)

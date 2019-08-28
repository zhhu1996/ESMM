import tensorflow as tf
import xdl

reader = xdl.DataReader("r1", # name of reader
                        paths=["./data.txt"], # file paths
                        enable_state=False) # enable reader state

reader.epochs(1).threads(1).batch_size(10).label_count(1)
reader.feature(name='sparse0', type=xdl.features.sparse)\
    .feature(name='sparse1', type=xdl.features.sparse)\
    .feature(name='deep0', type=xdl.features.dense, nvec=256)
reader.startup()

def train():
    batch = reader.read()
    sess = xdl.TrainSession()
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')
    emb2 = xdl.embedding('emb2', batch['sparse1'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')
    loss = model(batch['deep0'], [emb1, emb2], batch['label'])
    train_op = xdl.SGD(0.5).optimize()
    log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)
    sess = xdl.TrainSession(hooks=[log_hook])
    while not sess.should_stop():
        sess.run(train_op)

@xdl.tf_wrapper()
def model(deep, embeddings, labels):
    input = tf.concat([deep] + embeddings, 1)
    fc1 = tf.layers.dense(
        input, 256, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    fc2 = tf.layers.dense(
        fc1, 128, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    fc3 = tf.layers.dense(
        fc2, 64, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    fc4 = tf.layers.dense(
	fc3, 32, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    y = tf.layers.dense(
        fc4, 1, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    loss = tf.losses.sigmoid_cross_entropy(labels, y)
    return loss

train()


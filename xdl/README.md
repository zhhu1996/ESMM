# base NN

## 1. Dataset

本例采用的数据集是[data.txt](./data.txt)，这是它的一条样本记录

```
# sample_id  |  group_id  |  sparse  |  dense  |  label  |  timestamp
skey9|gkey9|sparse0@132:0.811190,201:0.836239,443:0.367191,243:0.872249;sparse1@1009:0.210185,560:0.681313,957:0.744213,677:0.839408,421:0.949919,661:0.554972,1009:0.524353,488:0.780305,705:0.199532,453:0.924493,40:0.213271|deep0@0.534764,0.317487,0.125192,0.423696,0.505640,0.895405,0.639840,0.562507,0.608992,0.830164,0.020030,0.2366480.587168|0|1544094735
```

数据集的格式规定如下：

| field     | desc                                       | value                                                        | example                                        |
| --------- | ------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------- |
| sample id | sample的唯一描述，用于调试                 | string                                                       | 7859345_420968_1007                            |
| group id  | 样本组的标识，连续一样的会聚合到一个样本组 | string                                                       | user_3423487                                   |
| sparse    | 稀疏特征，用kv表示                         | 多组特征用';'分隔，一个特征名字和内存用'@'分隔，内容多值用','分隔，稀疏的key和value用':'分隔 | clk_14@32490:1.0,32988:2.0;prefer@323423,32342 |
| dense     | 稠密特征                                   | 多组特征用';'分隔，一个特征内多值用','分隔                   | qscore@0.8,0.5;ad_price@33.8                   |
| label     | 目标                                       | float, 多值用','分隔                                         | 0.0,1.0                                        |
| ts        | 时间戳                                     | int                                                          | 1544094136                                     |

## 2. Demo

在xdl框架的基础上实现了一个5层nn的embedding模型，关于xdl的使用，点击这里[使用指南](https://github.com/alibaba/x-deeplearning/wiki/用户文档)

```python
import tensorflow as tf
import xdl

reader = xdl.DataReader("reader1", # name of reader
                        paths=["./data.txt"], # file paths
                        enable_state=False) # enable reader state

# read data
reader.epochs(1).threads(1).batch_size(10).label_count(1)
reader.feature(name='sparse0', type=xdl.features.sparse)\
    .feature(name='sparse1', type=xdl.features.sparse)\
    .feature(name='deep0', type=xdl.features.dense, nvec=256)
reader.startup()

# the pipeline
def train():
    batch = reader.read()
    sess = xdl.TrainSession()
    # embedding layer
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')
    emb2 = xdl.embedding('emb2', batch['sparse1'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')
    # cal loss
    loss = model(batch['deep0'], [emb1, emb2], batch['label'])
    # optimizer
    train_op = xdl.SGD(0.5).optimize()
    # logger
    log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)
    # train
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
```

## 3. Train

- 安装docker环境

```bash
sudo docker pull registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12
```

- 挂载目录

```bash
sudo docker run -v [path_to_xdl_code]:/home/xxx/deepctr -it registry.cn-hangzhou.aliyuncs.com/xdl/xdl:ubuntu-cpu-tf1.12 /bin/bash
```

- 进入训练目录

```bash
cd /home/xxx/deepctr
```

- 开始单机训练

```bash
python embedding_nn.py --run_mode=local
```

**注：首先要安装docker，之后再将xdl的docker环境pull到本机，最后将代码挂载到docker的目录中去执行**

- 结果

```
data parallel for file:  ./data.txt
('data paths:', ['./data.txt'])
loss:0.6982727
loss:0.7064079
loss:0.7008349
loss:0.69770515
loss:0.6966163
...
```


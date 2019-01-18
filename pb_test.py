

"""
利用pb模型对测试文件进行测试，这是使用c++调用tensorflow的pb模型的第一步
要点：tensorflow要一个batch一个batch传入
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
import os
import numpy as np
from sklearn import metrics
import time
from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def freeze_graph_test(pb_path, test_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试文本的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            input_x = sess.graph.get_tensor_by_name("input_x:0") ####这就是刚才取名的原因
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            
            # 定义输出的张量名称
            out_label = sess.graph.get_tensor_by_name("score/output:0")

            # 读取测试数据
            print("Loading test data...")
            start_time = time.time()

            x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, 600)

            batch_size = 128
            data_len = len(x_test)
            num_batch = int((data_len - 1) / batch_size) + 1

            y_test_cls = np.argmax(y_test, 1)
            y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果


            for i in range(num_batch):  # 逐批次处理
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                feed_dict = {
                    input_x: x_test[start_id:end_id],
                    input_keep_prob_tensor: 1.0
                }
                y_pred_cls[start_id:end_id] = sess.run(out_label, feed_dict=feed_dict)

            # 评估
            print("Precision, Recall and F1-Score...")
            print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

            # 混淆矩阵
            print("Confusion Matrix...")
            cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
            print(cm)

            time_dif = get_time_dif(start_time)
            print("Time usage:", time_dif)



if __name__ == '__main__':
    base_dir = 'data/cnews'
    out_pb_path = "checkpoints/frozen_model.pb"
    test_dir = "data/cnews/cnews.test.txt"
    vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    freeze_graph_test(pb_path=out_pb_path,test_path=test_dir)
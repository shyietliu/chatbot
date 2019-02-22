import tensorflow as tf
import csv


class DataProvider(object):
    def __init__(self):
        self.file_name = "data"
        self.feature_dict = {
            'feature': tf.FixedLenFeature([4], tf.float32),
            'label': tf.FixedLenFeature([], tf.float32)
        }
        self.batch_size = None
        self.drop_remainder = None

    def read_data(self, data_path):
        raw_data = []
        labels = []
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i != 0:
                    raw_data.append([float(ele) for ele in row[0:4]])
                    labels.append(int(row[4]))
        return raw_data, labels

    def convert2record(self, raw_data):
        def get_tfrecords_example(feature, label):
            tfrecords_features = {}
            tfrecords_features['feature'] = tf.train.Feature(float_list=tf.train.FloatList(value=feature))
            tfrecords_features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
            return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))

        def make_tfrecord(data, output_file_name):
            (feats, labels) = data
            output_file_name += '.tfrecord'
            tfrecord_wrt = tf.python_io.TFRecordWriter(output_file_name)
            ndatas = len(labels)
            for inx in range(ndatas):
                exmp = get_tfrecords_example(feats[inx], labels[inx])
                exmp_serial = exmp.SerializeToString()
                tfrecord_wrt.write(exmp_serial)
            tfrecord_wrt.close()

        make_tfrecord(raw_data, output_file_name=self.file_name)

    def parser(self, record, feature_dict):
        example = tf.parse_single_example(record, feature_dict)
        return example

    def input_fn_builder(self):
        def input_fn():
            data = tf.data.TFRecordDataset(self.file_name)

            data = data.apply(
                tf.contrib.data.map_and_batch(lambda record: self.parser(record, self.feature_dict),
                                              batch_size=self.batch_size,
                                              drop_remainder=self.drop_remainder))

            return data
        return input_fn()


if __name__ == '__main__':
    data_provider = DataProvider()
    data = data_provider.read_data("/Users/didi/PycharmProjects/chatbot/dataset/iris_training.csv")
    data_provider.convert2record(data)

    dataset = tf.data.TFRecordDataset('/Users/didi/PycharmProjects/chatbot/model/data.tfrecord')

    parsed_data = dataset.map(lambda record: data_provider.parser(record, data_provider.feature_dict)).batch(5)
    # parsed_data = dataset.apply(
    #     tf.contrib.data.map_and_batch(lambda record: data_provider.parser(record, data_provider.feature_dict),
    #                                   batch_size=5))
    iterator = parsed_data.make_one_shot_iterator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(iterator.get_next()))

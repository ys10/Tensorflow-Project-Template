# coding=utf-8
import tensorflow as tf


class DataSetLoader(object):
    def __init__(self, config, generator):
        self.config = config
        self.generator = generator
        with tf.variable_scope("data"):
            self.data_set = self.get_data_set_from_generator(self.generator.next, epochs=self.config.epochs,
                                                         batch_size=self.config.batch_size)
            self.iterator = self.data_set.make_one_shot_iterator()
            features, labels = self.iterator.get_next()
            self.next_data = {'features': features, 'labels': labels}
            self.data_set_init_op = self.iterator.make_initializer(self.data_set)

    @staticmethod
    def get_data_set_from_generator(generator_func, epochs=1, batch_size=16):
        data_set = tf.data.Dataset.from_generator(generator_func,
                                                  output_types=(tf.int32, tf.int32),
                                                  output_shapes=(tf.TensorShape([64]), tf.TensorShape([1])))
        data_set = data_set.repeat(epochs)
        data_set = data_set.batch(batch_size)
        return data_set

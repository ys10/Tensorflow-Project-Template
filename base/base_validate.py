# coding=utf-8
import tensorflow as tf


class BaseValidate:
    def __init__(self, sess, model, config, logger):
        self.sess = sess
        self.model = model
        self.config = config
        self.logger = logger
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def validate(self):
        """
        Predict whole data set(one epoch).
        """
        tf.logging.info('Validating...')
        self.validate_epoch()

    def validate_epoch(self):
        """
        Implement the logic of epoch:
        -loop over the number of iterations in the config and call the validate step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def validate_step(self):
        """
        Implement the logic of the validate step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

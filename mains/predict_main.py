# coding=utf-8
import tensorflow as tf
from models.fc_net_model import FCNetModel
from data_loader.data_set_loader import DataSetLoader
from data_loader.data_generator import DataGenerator
from utils.configs import process_config
from utils.logger import Logger
from operators.example_predictor import ExamplePredictor


def predict():
    predict_config = process_config("configs/predict.json")
    g = tf.Graph()
    with g.as_default():
        # load data
        predict_data_gen = DataGenerator()
        predict_data_loader = DataSetLoader(predict_config, predict_data_gen.next)
        next_data = predict_data_loader.next_data
        # create an instance of the model you want
        model = FCNetModel(predict_config, next_data)
        with tf.Session() as sess:
            # initialize data set
            sess.run([predict_data_loader.data_set_init_op])
            # create tensorboard logger
            logger = Logger(sess, predict_config)
            # create predictor and pass all the previous components to it
            predictor = ExamplePredictor(sess, model, predict_config, logger)
            # load model if exists
            model.load(sess)
            # here you use your model to predict
            predictor.predict()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    '''predict'''
    predict()
    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()

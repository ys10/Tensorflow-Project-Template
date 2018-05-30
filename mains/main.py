# coding=utf-8
import tensorflow as tf
from models.fc_net_model import FCNetModel
from data_loader.data_set_loader import DataSetLoader
from data_loader.data_generator import DataGenerator
from utils.configs import process_config
from utils.logger import Logger
from operators.example_trainer import ExampleTrainer
from operators.example_predictor import ExamplePredictor


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    train_config = process_config("configs/train.json")
    predict_config = process_config("configs/predict.json")
    train_data_gen = DataGenerator(train_config)
    predict_data_gen = DataGenerator(predict_config)
    train_data_loader = DataSetLoader(train_config, train_data_gen.next())
    predict_data_loader = DataSetLoader(predict_config, predict_data_gen.next())

    with tf.Session() as sess:
        # init model
        sess.run([train_data_loader.data_set_init_op])
        next_data = train_data_loader.next_data
        # create an instance of the model you want
        model = FCNetModel(train_config, next_data)
        # create tensorboard logger
        logger = Logger(sess, train_config)
        # create trainer and pass all the previous components to it
        trainer = ExampleTrainer(sess, model, train_config, logger)
        # load model if exists
        model.load(sess)
        # here you train your model
        trainer.train()
        # create predictor and pass all the previous components to it
        predictor = ExamplePredictor(sess, model, predict_config, logger)
        # reinitialize data set
        sess.run([predict_data_loader.data_set_init_op])
        next_data = predict_data_loader.next_data
        model.reset_data(next_data)
        # here you use your model to predict
        predictor.predict()

    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()

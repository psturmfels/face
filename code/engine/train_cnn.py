import tensorflow as tf
from utils.config import get, save_model
from data_scripts.fer2013_dataset import read_data_sets
from model.build_cnn import cnn

def get_weights(saver, sess):
    ''' load model weights if they were saved previously '''
    if is_file_prefix('TRAIN.CNN.CHECKPOINT'):
        saver.restore(sess, get('TRAIN.CNN.CHECKPOINT'))
        print('Yay! I restored weights from a saved model!')
    else:
        print('OK, I did not find a saved model, so I will start training from scratch!')

def report_training_progress(batch_index, input_layer, loss_func, validationSet, accuracy):
    ''' Update user on training progress '''
    if batch_index % 5:
        return
    print('starting batch number %d \033[100D\033[1A' % batch_index)
    if batch_index % 50:
        return
    error = loss_func.eval(feed_dict={input_layer: validationSet.images, true_labels: validationSet.labels})
    acc = accuracy.eval(feed_dict={input_layer: validationSet.images, true_labels: validationSet.labels})
    print('\n \t cross_entropy is about %f' % error)
    print(' \t accuracy is about %f' % acc)


def train_cnn(input_layer, prediction_layer, loss_func, optimizer, trainingSet, validationSet, accuracy):
    ''' Train CNN '''
    try:
        for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
            report_training_progress(
                batch_index, input_layer, loss_func, validationSet, accuracy)
            batch_images, batch_labels = trainingSet.next_batch(
                get('TRAIN.CNN.BATCH_SIZE'))
            optimizer.run(
                feed_dict={input_layer: batch_images, true_labels: batch_labels})
    except KeyboardInterrupt:
        print('OK, I will stop training even though I am not finished.')

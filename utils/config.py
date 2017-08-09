import tensorflow as tf

import glob

try:
    with open('config.json') as f:
        config = eval(f.read())
except SyntaxError:
    print('Uh oh... I could not parse the config file. Is it typed correctly? --- utils.config ')
except IOError:
    print('Uh oh... I could not find the config file. --- utils.config')


def get(attr, root=config):
    ''' Return value of specified configuration attribute. '''
    node = root
    for part in attr.split('.'):
        node = node[part]
    return node

def save_model(sess, path):
    saver = tf.train.Saver()
    save_path = saver.save(sess, path)
    print("Model saved to file: " + str(save_path))

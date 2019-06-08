#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

# flask
from flask import Flask,render_template,abort,request,send_from_directory
from datetime import timedelta

import json

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=timedelta(seconds=1)


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'test',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', 'models/289999.npy',
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv=None):
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    with tf.Session() as sess:
#         if FLAGS.phase == 'train':
#             # training phase
#             data = prepare_train_data(config)
#             model = CaptionGenerator(config)
#             sess.run(tf.global_variables_initializer())
#             if FLAGS.load:
#                 model.load(sess, FLAGS.model_file)
#             if FLAGS.load_cnn:
#                 model.load_cnn(sess, FLAGS.cnn_model_file)
#             tf.get_default_graph().finalize()
#             model.train(sess, data)

#         elif FLAGS.phase == 'eval':
#             # evaluation phase
#             coco, data, vocabulary = prepare_eval_data(config)
#             model = CaptionGenerator(config)
#             model.load(sess, FLAGS.model_file)
#             tf.get_default_graph().finalize()
#             model.eval(sess, coco, data, vocabulary)

#         else:
            # testing phase
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            results, img_results = model.test(sess, data, vocabulary)
            return results, img_results


@app.route('/imagecaption/do', methods=['POST'])
def imagecaption():
    #tf.app.run()
    # from tensorflow.python.platform import flags
    # f = flags.FLAGS
    # args = None
    # flags_passthrough = None
    try:
        results, img_results = main()
        print(results)
        return json.dumps({
            'status': 'ok',
            'resultInfo': results.to_json(),
            'img_results': json.dumps(img_results)
            })
    except Exception as e:
        print(e)
        return json.dumps({
            'status': 'error',
            'resultInfo': str(e),
            'img_results': ""
            })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

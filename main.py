import os

import pprint
import tensorflow as tf

from data import read_data, pad_data
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 20, "internal state dimension [20]")
flags.DEFINE_integer("nhop", 3, "number of hops [3]")
flags.DEFINE_integer("mem_size", 50, "maximum number of sentences that can be encoded into memory [50]")
flags.DEFINE_integer("batch_size", 32, "batch size to use during training [32]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_integer("anneal_epoch", 25, "anneal the learning rate every <anneal_epoch> epochs [25]")
flags.DEFINE_integer("babi_task", 1, "index of bAbI task for the network to learn [1]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("anneal_rate", 0.5, "learning rate annealing rate [0.5]")
flags.DEFINE_float("init_mean", 0., "weight initialization mean [0.]")
flags.DEFINE_float("init_std", 0.1, "weight initialization std [0.1]")
flags.DEFINE_float("max_grad_norm", 40, "clip gradients to this norm [40]")
flags.DEFINE_string("data_dir", "./bAbI/en-valid", "dataset directory [./bAbI/en_valid]")
flags.DEFINE_string("checkpoint_dir", "./checkpoints", "checkpoint directory [./checkpoints]")
flags.DEFINE_boolean("lin_start", False, "True for linear start training, False for otherwise [False]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for training [False]")
flags.DEFINE_boolean("show_progress", False, "print progress [False]")

FLAGS = flags.FLAGS


def main(_):
    word2idx = {}
    max_words = 0
    max_sentences = 0

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    train_stories, train_questions, max_words, max_sentences = read_data('{}/qa{}_train.txt'.format(FLAGS.data_dir, FLAGS.babi_task), word2idx, max_words, max_sentences)
    valid_stories, valid_questions, max_words, max_sentences = read_data('{}/qa{}_valid.txt'.format(FLAGS.data_dir, FLAGS.babi_task), word2idx, max_words, max_sentences)
    test_stories, test_questions, max_words, max_sentences = read_data('{}/qa{}_test.txt'.format(FLAGS.data_dir, FLAGS.babi_task), word2idx, max_words, max_sentences)

    pad_data(train_stories, train_questions, max_words, max_sentences)
    pad_data(valid_stories, valid_questions, max_words, max_sentences)
    pad_data(test_stories, test_questions, max_words, max_sentences)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    FLAGS.nwords = len(word2idx)
    FLAGS.max_words = max_words
    FLAGS.max_sentences = max_sentences
    
    pp.pprint(flags.FLAGS.__flags)
    
    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()
        
        if FLAGS.is_test:
            model.run(valid_stories, valid_questions, test_stories, test_questions)
        else:
            model.run(train_stories, train_questions, valid_stories, valid_questions)


if __name__ == '__main__':
    tf.app.run()

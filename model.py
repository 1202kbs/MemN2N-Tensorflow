import os
import math
import random

import numpy as np
import tensorflow as tf

from utils import ProgressBar


class MemN2N(object):
    
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.max_words = config.max_words
        self.max_sentences = config.max_sentences
        self.init_mean = config.init_mean
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.anneal_epoch = config.anneal_epoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.max_grad_norm = config.max_grad_norm
        
        self.lin_start = config.lin_start
        self.show_progress = config.show_progress
        self.is_test = config.is_test

        self.checkpoint_dir = config.checkpoint_dir
        
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        self.query = tf.placeholder(tf.int32, [None, self.max_words], name='input')
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name='time')
        self.target = tf.placeholder(tf.float32, [None, self.nwords], name='target')
        self.context = tf.placeholder(tf.int32, [None, self.mem_size, self.max_words], name='context')
        
        self.hid = []
        
        self.lr = None
        
        if self.lin_start:
            self.current_lr = 0.005
        else:
            self.current_lr = config.init_lr

        self.anneal_rate = config.anneal_rate
        self.loss = None
        self.optim = None
        
        self.sess = sess
        self.log_loss = []
        self.log_perp = []
    
    def build_memory(self):
        self.global_step = tf.Variable(0, name='global_step')
        
        zeros = tf.constant(0, tf.float32, [1, self.edim])
        self.A_ = tf.Variable(tf.random_normal([self.nwords - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        self.B_ = tf.Variable(tf.random_normal([self.nwords - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        self.C_ = tf.Variable(tf.random_normal([self.nwords - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        
        A = tf.concat([zeros, self.A_], axis=0)
        B = tf.concat([zeros, self.B_], axis=0)
        C = tf.concat([zeros, self.C_], axis=0)
        
        self.T_A_ = tf.Variable(tf.random_normal([self.mem_size - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        self.T_C_ = tf.Variable(tf.random_normal([self.mem_size - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        
        T_A = tf.concat([zeros, self.T_A_], axis=0)
        T_C = tf.concat([zeros, self.T_C_], axis=0)
        
        A_ebd = tf.nn.embedding_lookup(A, self.context)   # [batch_size, mem_size, max_length, edim]
        A_ebd = tf.reduce_sum(A_ebd, axis=2)              # [batch_size, mem_size, edim]
        T_A_ebd = tf.nn.embedding_lookup(T_A, self.time)  # [batch_size, mem_size, edim]
        A_in = tf.add(A_ebd, T_A_ebd)                     # [batch_size, mem_size, edim]
        
        C_ebd = tf.nn.embedding_lookup(C, self.context)   # [batch_size, mem_size, max_length, edim]
        C_ebd = tf.reduce_sum(C_ebd, axis=2)              # [batch_size, mem_size, edim]
        T_C_ebd = tf.nn.embedding_lookup(T_C, self.time)  # [batch_size, mem_size, edim]
        C_in = tf.add(C_ebd, T_C_ebd)                     # [batch_size, mem_size, edim]
        
        query_ebd = tf.nn.embedding_lookup(B, self.query) # [batch_size, max_length, edim]
        query_ebd = tf.reduce_sum(query_ebd, axis=1)      # [batch_size, edim]
        self.hid.append(query_ebd)
        
        for h in range(self.nhop):
            q3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim]) # [batch_size, edim] ==> [batch_size, 1, edim]
            p3dim = tf.matmul(q3dim, A_in, transpose_b=True)     # [batch_size, 1, edim] X [batch_size, edim, mem_size]
            p2dim = tf.reshape(p3dim, [-1, self.mem_size])       # [batch_size, mem_size]
            
            # If linear start, remove softmax layers
            if self.lin_start:
                p = p2dim
            else:
                p = tf.nn.softmax(p2dim)
            
            p3dim = tf.reshape(p, [-1, 1, self.mem_size]) # [batch_size, 1, mem_size]
            o3dim = tf.matmul(p3dim, C_in)                # [batch_size, 1, mem_size] X [batch_size, mem_size, edim]
            o2dim = tf.reshape(o3dim, [-1, self.edim])    # [batch_size, edim]
            
            a = tf.add(o2dim, self.hid[-1]) # [batch_size, edim]
            self.hid.append(a)              # [input, a_1, a_2, ..., a_nhop]
    
    def build_model(self):
        self.build_memory()
        
        self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], mean=self.init_mean, stddev=self.init_std))
        a_hat = tf.matmul(self.hid[-1], self.W)
        
        self.hypothesis = tf.nn.softmax(a_hat)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=a_hat, labels=self.target)
        
        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)
        
        params = [self.A_, self.B_, self.C_, self.T_A_, self.T_C_, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]
        
        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)
        
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()


    def train(self, train_stories, train_questions):
        N = int(math.ceil(len(train_questions) / self.batch_size))
        cost = 0
        
        if self.show_progress:
            bar = ProgressBar('Train', max=N)
        
        for idx in range(N):
            
            if self.show_progress:
                bar.next()
            
            if idx == N - 1:
                iterations = len(train_questions) - (N - 1) * self.batch_size
            else:
                iterations = self.batch_size
            
            query = np.ndarray([iterations, self.max_words], dtype=np.int32)
            time = np.zeros([iterations, self.mem_size], dtype=np.int32)
            target = np.zeros([iterations, self.nwords], dtype=np.float32)
            context = np.ndarray([iterations, self.mem_size, self.max_words], dtype=np.int32)
            
            for b in range(iterations):
                m = idx * self.batch_size + b
                
                curr_q = train_questions[m]
                q_text = curr_q['question']
                story_ind = curr_q['story_index']
                sent_ind = curr_q['sentence_index']
                answer = curr_q['answer'][0]
                
                curr_s = train_stories[story_ind]
                curr_c = curr_s[:sent_ind + 1]

                if len(curr_c) >= self.mem_size:
                    curr_c = curr_c[-self.mem_size:]
                    
                    for t in range(self.mem_size):
                        time[b, t].fill(t)
                else:
                    
                    for t in range(len(curr_c)):
                        time[b, t].fill(t)
                    
                    while len(curr_c) < self.mem_size:
                        curr_c.append([0.] * self.max_words)
                
                query[b, :] = q_text
                target[b, answer] = 1
                context[b, :, :] = curr_c

            _, loss, self.step = self.sess.run([self.optim, self.loss, self.global_step],
                                               feed_dict={self.query: query, self.time: time,
                                                          self.target: target, self.context: context})
            cost += np.sum(loss)
        
        if self.show_progress:
            bar.finish()
        
        return cost / len(train_questions)
    
    
    def test(self, test_stories, test_questions, label='Test'):
        N = int(math.ceil(len(test_questions) / self.batch_size))
        cost = 0
        
        if self.show_progress:
            bar = ProgressBar('Train', max=N)
        
        for idx in range(N):
            
            if self.show_progress:
                bar.next()
            
            if idx == N - 1:
                iterations = len(test_questions) - (N - 1) * self.batch_size
            else:
                iterations = self.batch_size
            
            query = np.ndarray([iterations, self.max_words], dtype=np.int32)
            time = np.zeros([iterations, self.mem_size], dtype=np.int32)
            target = np.zeros([iterations, self.nwords], dtype=np.float32)
            context = np.ndarray([iterations, self.mem_size, self.max_words], dtype=np.int32)
            
            for b in range(iterations):
                m = idx * self.batch_size + b
                
                curr_q = test_questions[m]
                q_text = curr_q['question']
                story_ind = curr_q['story_index']
                sent_ind = curr_q['sentence_index']
                answer = curr_q['answer'][0]
                
                curr_s = test_stories[story_ind]
                curr_c = curr_s[:sent_ind + 1]
                
                if len(curr_c) >= self.mem_size:
                    curr_c = curr_c[-self.mem_size:]
                    
                    for t in range(self.mem_size):
                        time[b, t].fill(t)
                else:
                    
                    for t in range(len(curr_c)):
                        time[b, t].fill(t)
                    
                    while len(curr_c) < self.mem_size:
                        curr_c.append([0.] * self.max_words)
                
                query[b, :] = q_text
                target[b, answer] = 1
                context[b, :, :] = curr_c

            _, loss, self.step = self.sess.run([self.optim, self.loss, self.global_step],
                                               feed_dict={self.query: query, self.time: time,
                                                          self.target: target, self.context: context})
            cost += np.sum(loss)
        
        if self.show_progress:
            bar.finish()
        
        return cost / len(test_questions)
    
    
    def run(self, train_stories, train_questions, test_stories, test_questions):
        if not self.is_test:

            for idx in range(self.nepoch):
                train_loss = np.sum(self.train(train_stories, train_questions))
                test_loss = np.sum(self.test(test_stories, test_questions, label='Validation'))
                
                self.log_loss.append([train_loss, test_loss])
                
                state = {
                    'loss': train_loss,
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'validation_loss': test_loss
                }
                
                print(state)
                
                
                # learning rate annealing
                if (not idx == 0) and (idx % self.anneal_epoch == 0):
                    self.current_lr = self.current_lr * self.anneal_rate
                    self.lr.assign(self.current_lr).eval()
            
                # If validation loss stops decreasing, insert softmax layers
                if idx == 0:
                    pass
                else:
                    if self.log_loss[idx][1] > self.log_loss[idx - 1][1]:
                        self.lin_start = False

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step=self.step.astype(int))
        else:
            self.load()
            
            valid_loss = np.sum(self.test(train_stories, train_questions, label='Validation'))
            test_loss = np.sum(self.test(test_stories, test_questions, label='Test'))
            
            state = {
                'validation_loss': valid_loss,
                'test_loss': test_loss
            }
            
            print(state)


    def predict(self, test_stories, test_questions):
        self.load()

        num_instances = len(test_questions)

        query = np.ndarray([num_instances, self.max_words], dtype=np.int32)
        time = np.zeros([num_instances, self.mem_size], dtype=np.int32)
        target = np.zeros([num_instances, self.nwords], dtype=np.float32)
        context = np.ndarray([num_instances, self.mem_size, self.max_words], dtype=np.int32)

        for b in range(num_instances):
            
            curr_q = test_questions[b]
            q_text = curr_q['question']
            story_ind = curr_q['story_index']
            sent_ind = curr_q['sentence_index']
            answer = curr_q['answer'][0]
            
            curr_s = test_stories[story_ind]
            curr_c = curr_s[:sent_ind + 1]
            
            if len(curr_c) >= self.mem_size:
                curr_c = curr_c[-self.mem_size:]
                
                for t in range(self.mem_size):
                    time[b, t].fill(t)
            else:
                
                for t in range(len(curr_c)):
                    time[b, t].fill(t)
                
                while len(curr_c) < self.mem_size:
                    curr_c.append([0.] * self.max_words)
            
            query[b, :] = q_text
            target[b, answer] = 1
            context[b, :, :] = curr_c

        predictions = self.sess.run(self.hypothesis, feed_dict={self.query: query, self.time: time, self.context: context})

        return predictions, target


        
    def load(self):
        print(' [*] Reading checkpoints...')
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] No checkpoint found")
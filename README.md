End-To-End Memory Networks in Tensorflow
========================================

Tensorflow implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4) for synthetic question and answering experiments (see Section 4). The original torch code from Facebook can be found [here](https://github.com/facebook/MemNN/tree/master/MemN2N-lang-model).

![alt tag](http://i.imgur.com/nv89JLc.png)


Prerequisites
-------------

This code requires [Tensorflow](https://www.tensorflow.org/). The bAbI dataset is stored in the 'bAbI' directory. The original files can be downloaded from [here](https://research.fb.com/downloads/babi/).
    
If you want to use `--show_progress True` option, you need to install python package `progress`.

    $ pip install progress
    
Usage
-----

To train a model with 3 hops and memory size of 50, run the following command:

    $ python main.py --nhop 3 --mem_size 50

To see all training options, run:

    $ python main.py --help

which will print:

    usage: main.py [-h] [--edim EDIM] [--nhop NHOP] [--mem_size MEM_SIZE]
               [--batch_size BATCH_SIZE] [--nepoch NEPOCH]
               [--anneal_epoch ANNEAL_EPOCH] [--babi_task BABI_TASK]
               [--init_lr INIT_LR] [--anneal_rate ANNEAL_RATE]
               [--init_mean INIT_MEAN] [--init_std INIT_STD]
               [--max_grad_norm MAX_GRAD_NORM] [--data_dir DATA_DIR]
               [--checkpoint_dir CHECKPOINT_DIR] [--lin_start [LIN_START]]
               [--nolin_start] [--is_test [IS_TEST]] [--nois_test]
               [--show_progress [SHOW_PROGRESS]] [--noshow_progress]

    optional arguments:
      -h, --help              show this help message and exit
      --edim EDIM             internal state dimension [20]
      --nhop NHOP             number of hops [3]
      --mem_size MEM_SIZE     maximum number of sentences that can be encoded into
                              memory [50]
      --batch_size BATCH_SIZE
                              batch size to use during training [32]
      --nepoch NEPOCH         number of epoch to use during training [100]
      --anneal_epoch ANNEAL_EPOCH
                              anneal the learning rate every <anneal_epoch> epochs
                              [25]
      --babi_task BABI_TASK
                              index of bAbI task for the network to learn [1]
      --init_lr INIT_LR       initial learning rate [0.01]
      --anneal_rate ANNEAL_RATE
                              learning rate annealing rate [0.5]
      --init_mean INIT_MEAN
                              weight initialization mean [0.]
      --init_std INIT_STD     weight initialization std [0.1]
      --max_grad_norm MAX_GRAD_NORM
                              clip gradients to this norm [40]
      --data_dir DATA_DIR     dataset directory [./bAbI/en_valid]
      --checkpoint_dir CHECKPOINT_DIR
                              checkpoint directory [./checkpoints]
      --lin_start [LIN_START]
                              True for linear start training, False for otherwise [False]
      --nolin_start
      --is_test [IS_TEST]     True for testing, False for training [False]
      --nois_test
      --show_progress [SHOW_PROGRESS] print progress [False]
      --noshow_progress

(Optional) If you want to see a progress bar, install `progress` with `pip`:

    $ pip install progress
    $ python main.py --nhop 3 --mem_size 50 --show_progress True

After training is finished, you can test and validate with:

    $ python main.py --is_test True --show_progress True


Acknowledgements
----------------

Majority of the source code in model.py is from: Taehoon Kim / [@carpedm20](http://carpedm20.github.io/).

The functions for reading the bAbI dataset and feeding it to the graph was modified. Also, the variable and operation names were modified for the ease of comparing the source code and the original paper.
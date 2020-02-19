import tensorflow as tf
import tensorbayes as tb
from codebase.args import args
from codebase.datasets import PseudoData, get_info
from utils import delete_existing, save_value, save_model, visualize_results
import os
import sys
import numpy as np

def update_dict(M, feed_dict, src=None, trg=None, bs=100, noise=None):
    """Update feed_dict with new mini-batch

    M         - (TensorDict) the model
    feed_dict - (dict) tensorflow feed dict
    src       - (obj) source domain. Contains train/test Data obj
    trg       - (obj) target domain. Contains train/test Data obj
    bs        - (int) batch size
    noise     - (bool) sample noise or not
    """
    if src:
        src_x, src_y = src.train.next_batch(bs)
        feed_dict.update({M.src_x: src_x, M.src_y: src_y})

    if trg:
        trg_x, trg_y = trg.train.next_batch(bs)
        feed_dict.update({M.trg_x: trg_x, M.trg_y: trg_y})

    if noise:
        trg_z = np.random.uniform(-1., 1., size=[bs, 100])
        feed_dict.update({M.trg_z: trg_z})

def train(M, src=None, trg=None, has_disc=True, saver=None, model_name=None):
    """Main training function

    Creates log file, manages datasets, trains model

    M          - (TensorDict) the model
    src        - (obj) source domain. Contains train/test Data obj
    trg        - (obj) target domain. Contains train/test Data obj
    has_disc   - (bool) whether model requires a discriminator update
    saver      - (Saver) saves models during training
    model_name - (str) name of the model being run with relevant parms info
    """
    # Training settings
    iterep = 1000
    itersave = 20000
    n_epoch = 200
    epoch = 0
    feed_dict = {}

    # Create a log directory and FileWriter
    log_dir = os.path.join(args.logdir, model_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)

    # Create a directory to save generated images
    gen_img_path = os.path.join(args.gendir, model_name)
    delete_existing(gen_img_path)
    os.makedirs(gen_img_path)

    # Create a save directory
    if saver:
        model_dir = os.path.join('checkpoints', model_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)

    # Replace src domain with psuedolabeled trg
    if args.dirt > 0:
        print "Setting backup and updating backup model"
        src = PseudoData(args.trg, trg, M.teacher)
        M.sess.run(M.update_teacher)

        # Sanity check model
        print_list = []
        if src:
            save_value(M.fn_ema_acc, 'test/src_test_ema_1k',
                     src.test,  train_writer, 0, print_list, full=False)

        if trg:
            save_value(M.fn_ema_acc, 'test/trg_test_ema',
                     trg.test,  train_writer, 0, print_list)
            save_value(M.fn_ema_acc, 'test/trg_train_ema_1k',
                     trg.train, train_writer, 0, print_list, full=False)

        print print_list

    if src: get_info(args.src, src)
    if trg: get_info(args.trg, trg)
    print "Batch size:", args.bs
    print "Iterep:", iterep
    print "Total iterations:", n_epoch * iterep
    print "Log directory:", log_dir

    best_acc = -1.
    trg_acc = -1.
    for i in xrange(n_epoch * iterep):
        if has_disc:
            # Run discriminator optimizer
            update_dict(M, feed_dict, src, trg, args.bs)
            summary, _ = M.sess.run(M.ops_disc, feed_dict)
            train_writer.add_summary(summary, i + 1)

            # Run generator optimizer
            update_dict(M, feed_dict, None, trg, args.bs, noise=True)
            summary, _ = M.sess.run(M.ops_gen, feed_dict)
            train_writer.add_summary(summary, i + 1)

        # Run main optimizer
        update_dict(M, feed_dict, src, trg, args.bs, noise=True)
        summary, _ = M.sess.run(M.ops_main, feed_dict)
        train_writer.add_summary(summary, i + 1)
        train_writer.flush()

        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(epoch, i),
                                            display=args.run >= 999)

        # Update pseudolabeler
        if args.dirt and (i + 1) % args.dirt == 0:
            print "Updating teacher model"
            M.sess.run(M.update_teacher)

        if (i + 1) % iterep == 0:
            gen_imgs = M.sess.run(M.trg_gen_x, feed_dict)
            manifold_h = int(np.floor(np.sqrt(args.bs)))
            manifold_w = int(np.floor(np.sqrt(args.bs)))
            visualize_results(gen_imgs, [manifold_h, manifold_w],
                os.path.join(gen_img_path, 'epoch_{}.png'.format((i + 1)/iterep)))

        # Log end-of-epoch values
        if end_epoch:
            print_list = M.sess.run(M.ops_print, feed_dict)

            if src:
                save_value(M.fn_ema_acc, 'test/src_test_ema_1k',
                         src.test,  train_writer, i + 1, print_list, full=False)

            if trg:
                trg_acc = save_value(M.fn_ema_acc, 'test/trg_test_ema',
                         trg.test,  train_writer, i + 1, print_list)
                save_value(M.fn_ema_acc, 'test/trg_train_ema_1k',
                         trg.train, train_writer, i + 1, print_list, full=False)

            print_list += ['epoch', epoch]
            print print_list

        if saver and trg_acc > best_acc:
            print("Saving new best model")
            saver.save(M.sess, os.path.join(model_dir, 'model_best'))
            best_acc = trg_acc

    # Saving final model
    if saver:
        save_model(saver, M, model_dir, i + 1)

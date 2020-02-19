import importlib
import tensorflow as tf
import tensorbayes as tb
import numpy as np
from extra_layers import basic_accuracy, vat_loss
from codebase.args import args
from pprint import pprint
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorbayes.layers import placeholder, constant
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_xent
nn = importlib.import_module('codebase.models.nns.{}'.format(args.nn))

def gada():
    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=tb.growth_config()),
        src_x = placeholder((None, 32, 32, 3)),
        src_y = placeholder((None, args.Y)),
        trg_x = placeholder((None, 32, 32, 3)),
        trg_y = placeholder((None, args.Y)),
        trg_z = placeholder((None, 100)),
        test_x = placeholder((None, 32, 32, 3)),
        test_y = placeholder((None, args.Y)),
    ))

    # Supervised and conditional entropy minimization
    src_e = nn.classifier(T.src_x, phase=True, enc_phase=1, enc_trim=args.etrim)
    src_g = nn.classifier(src_e, phase=True, gen_trim=args.gtrim, gen_phase=1, enc_trim=args.etrim)
    src_p = nn.classifier(src_g, phase=True, gen_trim=args.gtrim)
    trg_e = nn.classifier(T.trg_x, phase=True, enc_phase=1, enc_trim=args.etrim, reuse=True, internal_update=True)
    trg_g = nn.classifier(trg_e, phase=True, gen_trim=args.gtrim, gen_phase=1, enc_trim=args.etrim, reuse=True, internal_update=True)
    trg_p = nn.classifier(trg_g, phase=True, gen_trim=args.gtrim, reuse=True, internal_update=True)

    loss_src_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_p))
    loss_trg_cent = tf.reduce_mean(softmax_xent_two(labels=trg_p, logits=trg_p)) if args.tw > 0 else constant(0)

    # Domain confusion
    if args.dw > 0 and args.dirt == 0:
        real_logit = nn.real_feature_discriminator(src_e, phase=True)
        fake_logit = nn.real_feature_discriminator(trg_e, phase=True, reuse=True)

        loss_disc = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit))
        loss_domain = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.zeros_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit))

    else:
        loss_disc = constant(0)
        loss_domain = constant(0)

    # Virtual adversarial training (turn off src in non-VADA phase)
    loss_src_vat = vat_loss(T.src_x, src_p, nn.classifier) if args.sw > 0 and args.dirt == 0 else constant(0)
    loss_trg_vat = vat_loss(T.trg_x, trg_p, nn.classifier) if args.tw > 0 else constant(0)

    # Generate images and process generated images
    trg_gen_x = nn.trg_generator(T.trg_z)
    trg_gen_e = nn.classifier(trg_gen_x, phase=True, enc_phase=1, enc_trim=args.etrim, reuse=True, internal_update=True)
    trg_gen_g = nn.classifier(trg_gen_e, phase=True, gen_trim=args.gtrim, gen_phase=1, enc_trim=args.etrim, reuse=True, internal_update=True)
    trg_gen_p = nn.classifier(trg_gen_g, phase=True, gen_trim=args.gtrim, reuse=True, internal_update=True)

    # Feature matching loss function for generator
    loss_trg_gen_fm = tf.reduce_mean(tf.square(tf.reduce_mean(trg_g, axis=0) - tf.reduce_mean(trg_gen_g, axis=0))) if args.dirt == 0 else constant(0)

    # Unsupervised loss function
    if args.dirt == 0:
        logit_real = tf.reduce_logsumexp(trg_p, axis=1)
        logit_fake = tf.reduce_logsumexp(trg_gen_p, axis=1)
        dis_loss_real = -0.5*tf.reduce_mean(logit_real) + 0.5*tf.reduce_mean(tf.nn.softplus(logit_real))
        dis_loss_fake = 0.5*tf.reduce_mean(tf.nn.softplus(logit_fake))
        loss_trg_usv = dis_loss_real + dis_loss_fake    # UnSuperVised loss function
    else:
        loss_trg_usv = constant(0)

    # Evaluation (EMA)
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    var_class = tf.get_collection('trainable_variables', 'class/')
    ema_op = ema.apply(var_class)
    ema_p = nn.classifier(T.test_x, enc_phase=1, enc_trim=0, phase=False, reuse=True, getter=tb.tfutils.get_getter(ema))

    # Teacher model (a back-up of EMA model)
    teacher_p = nn.classifier(T.test_x, enc_phase=1, enc_trim=0, phase=False, scope='teacher')
    var_main = tf.get_collection('variables', 'class/(?!.*ExponentialMovingAverage:0)')
    var_teacher = tf.get_collection('variables', 'teacher/(?!.*ExponentialMovingAverage:0)')
    teacher_assign_ops = []
    for t, m in zip(var_teacher, var_main):
        ave = ema.average(m)
        ave = ave if ave else m
        teacher_assign_ops += [tf.assign(t, ave)]
    update_teacher = tf.group(*teacher_assign_ops)
    teacher = tb.function(T.sess, [T.test_x], tf.nn.softmax(teacher_p))

    # Accuracies
    src_acc = basic_accuracy(T.src_y, src_p)
    trg_acc = basic_accuracy(T.trg_y, trg_p)
    ema_acc = basic_accuracy(T.test_y, ema_p)
    fn_ema_acc = tb.function(T.sess, [T.test_x, T.test_y], ema_acc)

    # Optimizer
    dw = constant(args.dw) if args.dirt == 0 else constant(0)
    cw = constant(1)       if args.dirt == 0 else constant(args.bw)
    sw = constant(args.sw) if args.dirt == 0 else constant(0)
    tw = constant(args.tw)
    uw = constant(args.uw) if args.dirt == 0 else constant(0)
    loss_main = (dw * loss_domain +
                 cw * loss_src_class +
                 sw * loss_src_vat +
                 tw * loss_trg_cent +
                 tw * loss_trg_vat +
                 uw * loss_trg_usv)
    var_main = tf.get_collection('trainable_variables', 'class')
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)
    train_main = tf.group(train_main, ema_op)

    # Optimizer for feature discriminator
    if args.dw > 0 and args.dirt == 0:
        var_disc = tf.get_collection('trainable_variables', 'disc_real')
        train_disc = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_disc, var_list=var_disc)
    else:
        train_disc = constant(0)

    # Optimizer for generators
    if args.dirt == 0:
        fmw = constant(1)
        loss_trg_gen = (fmw * loss_trg_gen_fm)
        var_trg_gen = tf.get_collection('trainable_variables', 'trg_gen')
        trg_gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='trg_gen')
        with tf.control_dependencies(trg_gen_update_ops):
            train_trg_gen = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_trg_gen, var_list=var_trg_gen)
        train_gen = train_trg_gen
    else:
        fmw = constant(0)
        train_gen = constant(0)

    # Summarizations
    summary_disc = [tf.summary.scalar('domain/loss_disc', loss_disc),]
    summary_main = [tf.summary.scalar('domain/loss_domain', loss_domain),
                    tf.summary.scalar('class/loss_src_class', loss_src_class),
                    tf.summary.scalar('class/loss_trg_cent', loss_trg_cent),
                    tf.summary.scalar('class/loss_trg_usv', loss_trg_usv),
                    tf.summary.scalar('lipschitz/loss_trg_vat', loss_trg_vat),
                    tf.summary.scalar('lipschitz/loss_src_vat', loss_src_vat),
                    tf.summary.scalar('hyper/dw', dw),
                    tf.summary.scalar('hyper/cw', cw),
                    tf.summary.scalar('hyper/sw', sw),
                    tf.summary.scalar('hyper/tw', tw),
                    tf.summary.scalar('hyper/uw', uw),
                    tf.summary.scalar('hyper/fmw', fmw),
                    tf.summary.scalar('acc/src_acc', src_acc),
                    tf.summary.scalar('acc/trg_acc', trg_acc)]
    summary_gen  = [tf.summary.scalar('gen/loss_trg_gen_fm', loss_trg_gen_fm),
                    tf.summary.image('gen/trg_gen_img', trg_gen_x),]

    # Merge summaries
    summary_disc = tf.summary.merge(summary_disc)
    summary_main = tf.summary.merge(summary_main)
    summary_gen  = tf.summary.merge(summary_gen)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('disc'), loss_disc,
                   c('domain'), loss_domain,
                   c('class'), loss_src_class,
                   c('cent'), loss_trg_cent,
                   c('trg_vat'), loss_trg_vat,
                   c('src_vat'), loss_src_vat,
                   c('src'), src_acc,
                   c('trg'), trg_acc]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_main = [summary_main, train_main]
    T.ops_gen  = [summary_gen , train_gen]
    T.fn_ema_acc = fn_ema_acc
    T.teacher = teacher
    T.update_teacher = update_teacher
    T.trg_gen_x = trg_gen_x
    T.trg_gen_p = trg_gen_p
    T.src_p = src_p
    T.trg_p = trg_p
    T.ema_p = ema_p

    return T

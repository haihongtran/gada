import os
import sys
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model',  type=str,   default='gada',    help="Model name")
parser.add_argument('--src',    type=str,   default='mnist',   help="Src data")
parser.add_argument('--trg',    type=str,   default='svhn',    help="Trg data")
parser.add_argument('--nn',     type=str,   default='small',   help="Architecture")
parser.add_argument('--gtrim',  type=int,   default=1,         help="Generator Trim")
parser.add_argument('--etrim',  type=int,   default=3,         help="Encoder Trim")
parser.add_argument('--inorm',  type=int,   default=1,         help="Instance normalization flag")
parser.add_argument('--radius', type=float, default=3.5,       help="Perturbation 2-norm ball radius")
parser.add_argument('--dw',     type=float, default=1e-2,      help="Domain weight")
parser.add_argument('--bw',     type=float, default=1e-2,      help="Beta (KL) weight")
parser.add_argument('--sw',     type=float, default=1,         help="Src weight")
parser.add_argument('--tw',     type=float, default=1e-2,      help="Trg weight")
parser.add_argument('--uw',     type=float, default=1.,        help="Unsupervised loss weight")
parser.add_argument('--lr',     type=float, default=2e-4,      help="Learning rate")
parser.add_argument('--bs',     type=int,   default=64,        help="Batch size")
parser.add_argument('--dirt',   type=int,   default=0,         help="0 == VADA, >0 == DIRT-T interval")
parser.add_argument('--run',    type=int,   default=999,       help="Run index. >= 999 == debugging")
parser.add_argument('--datadir',type=str,   default='data',    help="Data directory")
parser.add_argument('--logdir', type=str,   default='log',     help="Log directory")
parser.add_argument('--gendir', type=str,   default='genimgs', help="Generated images directory")
codebase_args.args = args = parser.parse_args()

# Argument overrides and additions
src2Y = {'mnist': 10, 'mnistm': 10, 'digit': 10, 'svhn': 10, 'cifar': 9, 'stl': 9, 'sign': 43}
args.Y = src2Y[args.src]
args.H = 32
args.bw = args.bw if args.dirt > 0 else 0.  # mask bw when training
pprint(vars(args))

from codebase.models.gada import gada
from codebase.train import train
from codebase.datasets import get_data

# Make model name
setup = [
    ('model={:s}',  args.model),
    ('src={:s}',    args.src),
    ('trg={:s}',    args.trg),
    ('nn={:s}',     args.nn),
    ('lr={:.0e}',   args.lr),
    ('inorm={:d}',  args.inorm),
    ('dw={:.0e}',   args.dw),
    ('bw={:.0e}',   args.bw),
    ('sw={:.0e}',   args.sw),
    ('tw={:.0e}',   args.tw),
    ('uw={:.0e}',   args.uw),
    ('dirt={:05d}', args.dirt),
    ('run={:04d}',  args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

M = gada()
M.sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if args.dirt > 0:
    run = args.run if args.run < 999 else 0
    setup = [
        ('model={:s}',  args.model),
        ('src={:s}',    args.src),
        ('trg={:s}',    args.trg),
        ('nn={:s}',     args.nn),
        ('lr={:.0e}',   args.lr),
        ('inorm={:d}',  args.inorm),
        ('dw={:.0e}',   args.dw),
        ('bw={:.0e}',   0),
        ('sw={:.0e}',   args.sw),
        ('tw={:.0e}',   args.tw),
        ('uw={:.0e}',   args.uw),
        ('dirt={:05d}', 0),
        ('run={:04d}',  args.run)
    ]
    init_model = '_'.join([t.format(v) for (t, v) in setup])
    print "Init model:", init_model
    model_path = os.path.join('checkpoints', init_model, 'model_best')
    if os.path.exists(model_path + '.index'):
        saver.restore(M.sess, model_path)
        print "Restored from {}".format(model_path)
    else:
        path = tf.train.latest_checkpoint(os.path.join('checkpoints', init_model))
        saver.restore(M.sess, path)
        print "Restored from {}".format(path)

src = get_data(args.src)
trg = get_data(args.trg)

train(M, src, trg,
      saver=saver,
      has_disc=args.dirt == 0,
      model_name=model_name)

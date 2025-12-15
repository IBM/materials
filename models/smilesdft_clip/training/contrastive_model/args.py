import argparse
import yaml
import re
from .utils import dotdict


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yaml')
    parser.add_argument('--args', help="learning rate", type=bool, required=False, default=False)

    # Train
    #train_arg = parser.add_argument_group('Train')
    parser.add_argument('--n_batch',
                           type=int, default=512,
                           help='Batch size')
    parser.add_argument('--unlike_alpha',
                           type=float, default=1.0,
                           help='unlikelihood loss alpha weight')
    parser.add_argument('--from_scratch',
                           action='store_true', default=False,
                           help='train on qm9 from scratch')
    parser.add_argument('--unlikelihood',
                           action='store_true', default=False,
                           help='use unlikelihood loss with gpt pretrain')
    parser.add_argument('--grad_acc',
                           type=int, default=1,
                           help='number of batches to accumulate gradients')
    parser.add_argument('--checkpoint_every',
                           type=int, default=1000,
                           help='save checkpoint every x iterations')
    parser.add_argument('--clip_grad',
                           type=int, default=50,
                           help='Clip gradients to this value')
    parser.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    parser.add_argument('--lr_end',
                           type=float, default=3 * 1e-4,
                           help='Maximum lr weight value')
    parser.add_argument('--lr_multiplier',
                           type=int, default=1,
                           help='lr weight multiplier')
    parser.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    parser.add_argument('--n_jobs',
                           type=int, default=1,
                           help='Number of threads')
    parser.add_argument('--accelerator',
                        type=str, default='ddp',
                        help='The accelerator backend to use (previously known as distributed_backend)')
    parser.add_argument('--num_nodes',
                        type=int, default=1,
                        help='number of GPU nodes for distributed training')
    parser.add_argument('--device',
                        type=str, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=12345,
                        help='Seed')
    parser.add_argument('--init_params_from',
                        type=str, default='',
                        help='Path to a ckpt used to initialize the parameters if no restart_path is provided')
    parser.add_argument('--train_decoder_every',
                        type=int, default=10,
                        help='Optimize decoder params every n batches')
    parser.add_argument('--lr_decoder',
                        type=float, default=1e-4,
                        help='Learning rate for decoder part')
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--save_checkpoint_path', default='/data', help='checkpoint saving path')
    parser.add_argument('--load_checkpoint_path', default='', help='checkpoint loading path')

    #common_arg = parser.add_argument_group('Common')
    parser.add_argument('--vocab_load',
                            type=str, required=False,
                            help='Where to load the vocab')
    parser.add_argument('--n_samples',
                            type=int, required=False,
                            help='Number of samples to sample')
    parser.add_argument('--gen_save',
                            type=str, required=False,
                            help='Where to save the gen molecules')
    parser.add_argument("--max_len",
                            type=int, default=100,
                            help="Max of length of SMILES")
    parser.add_argument('--train_load',
                            type=str, required=False,
                            help='Where to load the model')
    parser.add_argument('--val_load',
                            type=str, required=False,
                            help='Where to load the model')
    parser.add_argument('--n_workers',
                            type=int, required=False, default=1,
                            help='Where to load the model')
    #beam search hyper parameters
    parser.add_argument('--beam_size', type=int, default=0,
                            help="Number of beams to generate")
    parser.add_argument('--num_seq_returned', type=int, default=0,
                            help="number of beams to be returned (must be <= beam_size")
    parser.add_argument('--min_len', type=int, default=1,
                            help="minimum length to be generated")
    parser.add_argument('--nucleus_thresh', type=float, default=.9,
                            help="nucleus sampling threshold")
    parser.add_argument('--finetune_path',
                           type=str, default="",
                           help='path to  trainer file to continue training')
    parser.add_argument('--restart_path',
                           type=str, default="",
                           help='path to  trainer file to continue training')
    parser.add_argument('--data_path',
                           type=str, default="",
                           help='path to pubchem file')
    parser.add_argument('--pretext_size',
                           type=int, default=0,
                           help='number of k-mers to pretext')
    parser.add_argument('--model_save_dir',
                            type=str, required=False, default='./models_dump/',
                            help='Where to save the models/log/config/vocab')
    parser.add_argument('--model_save',
                            type=str, required=False, default='model.pt',
                            help='Where to save the model')
    #parser.add_argument('--save_frequency',
    #                        type=int, default=20,
    #                        help='How often to save the model')
    parser.add_argument('--num_epoch',
                            type=int, default=1,
                            help='number of epochs to train')
    #parser.add_argument('--num_iter',
    #                        type=int, default=-1,
    #                        help='how many itersations per epoch (for unlikelihood tuning)')
    parser.add_argument('--log_file',
                            type=str, required=False,
                            help='Where to save the log')
    parser.add_argument('--tb_loc',
                            type=str, required=False,
                            help='Where to save the tensorflow location')
    parser.add_argument('--config_save',
                            type=str, required=False,
                            help='Where to save the config')
    parser.add_argument('--vocab_save',
                            type=str,
                            help='Where to save the vocab')

   # resume_arg = parser.add_argument_group('Resume')
    parser.add_argument('--debug',
                           default=False, action='store_true',
                           help='do not erase cache at end of program')
    parser.add_argument('--fast_dev_run',
                           default=False,
                           help='This flag runs a “unit test” by running n if set to n (int) else 1 if set to True training and validation batch(es).')
    parser.add_argument('--freeze_model',
                           default=False, action='store_true',
                           help='freeze weights of bert model during fine tuning')
    parser.add_argument('--resume',
                           default=False, action='store_true',
                           help='Resume from a saved model')
    parser.add_argument('--rotate',
                           default=False, action='store_true',
                           help='use rotational relative embedding')
    parser.add_argument('--model_load',
                            type=str, required=False,
                            help='Where to load the model')
    parser.add_argument('--root_dir',
                            type=str, required=False, default='.',
                            help='location of root dir')
    parser.add_argument('--config_load',
                            type=str, required=False,
                            help='Where to load the config')
    parser.add_argument('--gpus',
                            type=int, required=False, default=1,
                            help='number of gpus to use')
    #parser.add_argument('--start_epoch',
    #                        type=int, required=False, default=0,
    #                        help='Where to load the config')

    parser.add_argument('--model_arch',
                            type=str, required=False,
                            help='used to teack model arch in params')
    parser.add_argument('--eval_every',
                            type=int, default=50000,
                            help='run evaluation every x iterations')
    parser.add_argument('--num_feats',
                            type=int, required=False, default=32,
                            help='number of random reatures for FAVOR+')
    parser.add_argument('--max_epochs',
                            type=int, required=False, default=1,
                            help='max number of epochs')

    # debug() FINE TUNEING
    parser.add_argument('--mode',
                           type=str, default='cls',
                           help='type of pooling to use')
    parser.add_argument("--dataset_length", type=int, default=None, required=False)
    parser.add_argument("--num_workers", type=int, default=0, required=False)
    parser.add_argument("--dropout", type=float, default=0.1, required=False)
    parser.add_argument("--dataset_name", type=str, required=False, default="sol")
    parser.add_argument("--measure_name", type=str, required=False, default="measure")

    # parser.add_argument("--use_bn", type=int, default=0)
    parser.add_argument("--use_linear", type=int, default=0)

    parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--weight_decay", type=float, default=5e-4)
    # parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)

    return parser


def get_yaml_loader():
    """Loads YAML loader handling scientific values."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return loader


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    yaml_loader = get_yaml_loader()

    if not args.args:  # args priority is higher than yaml
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml_loader)
        opt.update(args)
        args = dotdict(opt)
    else:  # yaml priority is higher than args
        opt = yaml.load(open(args.config), Loader=yaml_loader)
        opt.update(vars(args))
        args = dotdict(opt)

    return args
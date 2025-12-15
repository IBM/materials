import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Train
    parser.add_argument('--n_batch',
                           type=int, default=512,
                           help='Batch size')
    parser.add_argument('--checkpoint_every',
                           type=int, default=1000,
                           help='save checkpoint every x iterations')
    parser.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    parser.add_argument('--seed',
                        type=int, default=12345,
                        help='Seed')
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--save_checkpoint_path', type=str, required=False, default='/data', help='checkpoint saving path')
    parser.add_argument('--load_checkpoint_path', type=str, required=False, default='', help='checkpoint loading path')

    #common_arg = parser.add_argument_group('Common')
    parser.add_argument("--max_len",
                            type=int, default=100,
                            help="Max of length of SMILES")
    parser.add_argument('--n_workers',
                            type=int, required=False, default=1,
                            help='Where to load the model')
    parser.add_argument('--max_epochs',
                            type=int, required=False, default=1,
                            help='max number of epochs')
    parser.add_argument("--num_workers", type=int, default=0, required=False)
    parser.add_argument("--dropout", type=float, default=0.1, required=False)
    parser.add_argument("--dataset_name", type=str, required=False, default="sol")
    parser.add_argument("--measure_name", type=str, required=False, default="measure")
    parser.add_argument("--tokenizer_path", type=str, required=True, default="./str_bamba/tokenizer_bert_bpe.json")
    parser.add_argument("--config_path", type=str, required=True, default="../str_bamba/config/config_encoder-decoder.json")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--pubchem_files_path", type=str, required=True)
    parser.add_argument("--polymer_files_path", type=str, required=True)
    parser.add_argument("--formulation_files_path", type=str, required=True)
    parser.add_argument("--data_cache_dir", type=str, required=False, default=None)

    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
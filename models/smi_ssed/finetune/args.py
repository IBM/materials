import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    #model_arg = parser.add_argument_group('Model')
    parser.add_argument('--n_layer',
                           type=int, default=12,
                           help='Mamba number of layers')
    parser.add_argument(
        "--d_dropout", type=float, default=0.1, help="Decoder layers dropout"
    )
    parser.add_argument('--n_embd',
                           type=int, default=768,
                           help='Latent vector dimensionality')
    parser.add_argument('--dt_rank',
                           type=str, default='auto')
    parser.add_argument('--d_state',
                           type=int, default=16)
    parser.add_argument('--expand_factor',
                           type=int, default=2)
    parser.add_argument('--d_conv',
                           type=int, default=4)
    parser.add_argument('--dt_min',
                           type=float, default=0.001)
    parser.add_argument('--dt_max',
                           type=float, default=0.1)
    parser.add_argument('--dt_init',
                           type=str, default='random')
    parser.add_argument('--dt_scale',
                           type=float, default=1.0)
    parser.add_argument('--dt_init_floor',
                           type=float, default=1e-4)
    parser.add_argument('--bias',
                           type=int, default=0)
    parser.add_argument('--conv_bias',
                           type=int, default=1)


    # Train
    #train_arg = parser.add_argument_group('Train')
    parser.add_argument('--n_batch',
                           type=int, default=512,
                           help='Batch size')
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
    parser.add_argument('--device',
                        type=str, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=12345,
                        help='Seed')
    parser.add_argument('--lr_decoder',
                        type=float, default=1e-4,
                        help='Learning rate for decoder part')
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
    parser.add_argument('--max_epochs',
                            type=int, required=False, default=1,
                            help='max number of epochs')

    # debug() FINE TUNEING
    # parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--mode',
                           type=str, default='cls',
                           help='type of pooling to use')
    parser.add_argument("--dataset_length", type=int, default=None, required=False)
    parser.add_argument("--num_workers", type=int, default=0, required=False)
    parser.add_argument("--dropout", type=float, default=0.1, required=False)
    #parser.add_argument("--dims", type=int, nargs="*", default="", required=False)
    parser.add_argument(
        "--smiles_embedding",
        type=str,
        default="/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/embeddings/protein/ba_embeddings_tanh_512_2986138_2.pt",
    )
    # parser.add_argument("--train_pct", type=str, required=False, default="95")
    #parser.add_argument("--aug", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=False, default="sol")
    parser.add_argument("--measure_name", type=str, required=False, default="measure")
    #parser.add_argument("--emb_type", type=str, required=True)
    parser.add_argument("--checkpoints_folder", type=str, required=True)
    #parser.add_argument("--results_dir", type=str, required=True)
    #parser.add_argument("--patience_epochs", type=int, required=True)
    parser.add_argument("--model_path", type=str, default="./smi_ted/")
    parser.add_argument("--ckpt_filename", type=str, default="smi_ssed_130.pt")
    parser.add_argument("--restart_filename", type=str, default="")
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument("--save_every_epoch", type=int, default=0)
    parser.add_argument("--save_ckpt", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--smi_ssed_version", type=str, default="v1")
    parser.add_argument("--train_decoder", type=int, default=1)
    parser.add_argument("--target_metric", type=str, default="rmse")
    parser.add_argument("--loss_fn", type=str, default="mae")

    parser.add_argument(
        "--data_root",
        type=str,
        required=False,
        default="/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/affinity",
    )
    # parser.add_argument("--use_bn", type=int, default=0)
    parser.add_argument("--use_linear", type=int, default=0)

    parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--weight_decay", type=float, default=5e-4)
    # parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)

    return parser
def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args


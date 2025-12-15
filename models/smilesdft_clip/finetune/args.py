import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=False, default="")
    parser.add_argument("--grid_path", type=str, required=False, default="")

    parser.add_argument(
        "--lr_start", type=float, default=3 * 1e-4, help="Initial lr value"
    )
    parser.add_argument(
        "--max_epochs", type=int, required=False, default=1, help="max number of epochs"
    )
    
    parser.add_argument("--num_workers", type=int, default=0, required=False)
    parser.add_argument("--dropout", type=float, default=0.1, required=False)
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size")
    parser.add_argument("--dataset_name", type=str, required=False, default="sol")
    parser.add_argument("--measure_name", type=str, required=False, default="measure")
    parser.add_argument("--checkpoints_folder", type=str, required=False)
    parser.add_argument("--model_path", type=str, default="./smi_ted/")
    parser.add_argument("--ckpt_filename", type=str, default="smi_ted_Light_40.pt")
    parser.add_argument("--restart_filename", type=str, default="")
    parser.add_argument("--image_filename", type=str, default="")
    parser.add_argument("--text_filename", type=str, default="")
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument("--save_every_epoch", type=int, default=0)
    parser.add_argument("--save_ckpt", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--target_metric", type=str, default="rmse")
    parser.add_argument("--loss_fn", type=str, default="mae")
    parser.add_argument("--arch", type=str, default='clip')

    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
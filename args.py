import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data params ---
    parser.add_argument("--dataset", type=str.upper, default="SMAP")
    parser.add_argument("--graph", type=str, default='gat')
    parser.add_argument("--num", type=str, default="6", help="Required for ASD dataset. <group_index>") #两个不同数据集
    parser.add_argument("--group", type=str, default="1-8", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=60)#1
    parser.add_argument('--topK', type=int, default=5, help='k')#1
    parser.add_argument('--embed_size', type=int, default=25, help='embed_size')#嵌入维度的大小
    parser.add_argument("--epochs", type=int, default="1")#1
    parser.add_argument("--bs", type=int, default=128)#1 128->64
    parser.add_argument("--init_lr", type=float, default=0.002)#1  0.003->0.002
    parser.add_argument("--dropout", type=float, default=0.3)#1  0.4->0.3->0.2
    parser.add_argument("--level", type=float, default=0.85)#
    parser.add_argument("--q", type=float, default=0.05)#

    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False)

    # -- Model params ---
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=150)
    # Forecasting Model
    parser.add_argument("--fc_n_layers", type=int, default=2)#预测层2层
    parser.add_argument("--fc_hid_dim", type=int, default=150)#隐藏层150
    # Reconstruction Model
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--in_channels', type=int, default=1, help='输入通道信息，只有速度信息，所以通道为1')

    # --- Train params ---
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)

    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    # --- Other ---
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--threshold", type=int, default=3)

    return parser

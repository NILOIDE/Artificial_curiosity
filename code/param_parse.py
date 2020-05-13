import argparse
from utils.utils import CONV_LAYERS2014
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env_name', type=str, default='Breakout-v0')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--name', type=str, default='2dCuriosity')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--export_interval', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=20000)
    parser.add_argument('--buffer_size', type=int, default=int(5e4))
    parser.add_argument('--train_steps', type=int, default=int(5e6))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_static', type=bool, default=True)
    parser.add_argument('--eps_half', help='Epsilon at half t in exponential epsilon decay', type=float, default=0.08)
    parser.add_argument('--eps_min', help='Minimum clipped eps', type=float, default=0.01)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alg_target_net_steps', type=int, default=1000)
    parser.add_argument('--alg_soft_target', type=bool, default=False)
    parser.add_argument('--alg_lr', type=float, default=0.00001)
    parser.add_argument('--z_dim', type=tuple, default=(512,))
    parser.add_argument('--wm_target_net_steps', type=int, default=1000)
    parser.add_argument('--wm_soft_target', type=bool, default=False)
    parser.add_argument('--wm_lr', type=float, default=0.00001)
    parser.add_argument('--wm_tau', type=float, default=0.01)
    parser.add_argument('--wm_warmup_steps', type=int, default=0)
    parser.add_argument('--intr_rew_norm_type', type=str, default='max_history')

    parser.add_argument('--encoder_type', type=str, default="random", choices=["none", 'random', 'cont', 'idf', 'vae'])
    parser.add_argument('--decoder', type=bool, default=False)
    parser.add_argument('--resize_dim', type=tuple, default=(84, 84))
    parser.add_argument('--grayscale', type=bool, default=True)
    parser.add_argument('--frame_stack', type=int, default=4)
    parser.add_argument('--conv_layers', type=tuple, default=CONV_LAYERS2014)
    parser.add_argument('--stochastic_latent', type=bool, default=False)
    parser.add_argument('--encoder_batchnorm', type=bool, default=False)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--hinge_value', type=float, default=1.0)

    args = parser.parse_args().__dict__
    args['time_stamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return args

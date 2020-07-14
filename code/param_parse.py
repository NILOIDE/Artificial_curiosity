import argparse
from utils.utils import CONV_LAYERS2014, CONV_LAYERS2015
from datetime import datetime
from ast import literal_eval as make_tuple

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env_name', type=str, default='Breakout-v0')
    parser.add_argument('--save_dir', type=str, default='final_results/')
    parser.add_argument('--name', type=str, default='zdim32_hdim64_eps01_ns100_hinge01_uniform_lr-3_test')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--export_interval', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=int(2e4))
    parser.add_argument('--buffer_size', type=int, default=int(5e4))
    parser.add_argument('--train_steps', type=int, default=int(2e6))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_static', type=bool, default=True)
    parser.add_argument('--eps_half', help='Epsilon at half t in exponential epsilon decay', type=float, default=0.08)
    parser.add_argument('--eps_min', help='Minimum clipped eps', type=float, default=.1)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alg_target_net_steps', type=int, default=1000)
    parser.add_argument('--alg_soft_target', type=bool, default=False)
    parser.add_argument('--alg_lr', type=float, default=1e-4)
    parser.add_argument('--z_dim', type=str, default='(32,)')
    parser.add_argument('--wm_h_dim', type=str, default='(64,)')
    parser.add_argument('--wm_opt', type=str, default='sgd', choices=['adam', 'adam'])
    parser.add_argument('--wm_target_net_steps', type=int, default=0)
    parser.add_argument('--wm_soft_target', type=bool, default=False)
    parser.add_argument('--wm_lr', type=float, default=1e-3)
    parser.add_argument('--wm_enc_lr', type=float, default=1e-2)
    parser.add_argument('--wm_tau', type=float, default=0.01)
    parser.add_argument('--wm_warmup_steps', type=int, default=0)
    parser.add_argument('--intr_rew_norm_type', type=str, default='whiten_history',
                        choices=['none', 'max', 'whiten', 'max_history', 'whiten_history'])
    parser.add_argument('--intr_rew_mean_n', help='Length of history in running mean', type=int, default=1000)

    parser.add_argument('--encoder_type', type=str, default="cont",
                        choices=['tab', 'none', 'random', 'cont', 'idf', 'vae'])
    parser.add_argument('--decoder', type=bool, default=False)
    parser.add_argument('--resize_dim', type=str, default='(84, 84)')
    parser.add_argument('--grayscale', type=bool, default=True)
    parser.add_argument('--frame_stack', type=int, default=4)
    parser.add_argument('--conv_layers', type=tuple, default=CONV_LAYERS2015)
    parser.add_argument('--stochastic_latent', type=bool, default=False)
    parser.add_argument('--encoder_batchnorm', type=bool, default=False)

    parser.add_argument('--neg_samples', type=int, default=100)
    parser.add_argument('--hinge_value', type=float, default=0.1)
    parser.add_argument('--idf_inverse_hdim', type=str, default='(64,)')

    parser.add_argument('--gridworld_ns_pool', type=str, default="uniform", choices=['visited', 'uniform', 'visited_uniform'])

    args = parser.parse_args().__dict__
    args['time_stamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    # Convert to tuples
    args['z_dim'] = make_tuple(args['z_dim'])
    args['wm_h_dim'] = make_tuple(args['wm_h_dim'])
    args['resize_dim'] = make_tuple(args['resize_dim'])
    args['idf_inverse_hdim'] = make_tuple(args['idf_inverse_hdim'])
    return args

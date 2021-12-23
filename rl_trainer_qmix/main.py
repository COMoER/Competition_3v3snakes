import argparse
import datetime

from tensorboardX import SummaryWriter

from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
from trainer import rnn_trainer,step_trainer


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main(args):
    if args.step_model:
        step_trainer(args)
    else:
        rnn_trainer(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="qmix", type=str, help="")
    parser.add_argument('--max_episodes', default=150000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)

    parser.add_argument('--buffer_size', default=int(5e3), type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--update_target_episode', default=200, type=int,help="in step model,when using hard_update it means target per %step but not episodes")
    parser.add_argument('--DoubleDQN', action='store_true')
    parser.add_argument('--judgeIndependent', action='store_true')
    parser.add_argument('--mode', default=4, type=int,help = "one_hot conv/normal conv/fc/fixed 1/2/3/4")
    parser.add_argument('--compete',action='store_true')
    parser.add_argument('--well_enemy',action = 'store_true')
    parser.add_argument('--win_gain', default=30, type=int)
    parser.add_argument('--lose_gain', default=20, type=int)
    parser.add_argument('--step_radio', default=20., type=float)
    parser.add_argument("--step_model",action="store_true")
    parser.add_argument("--self_compete", action="store_true")
    parser.add_argument("--tau", default=1e-3, type=float)
    parser.add_argument("--delay_times",default = 5e4, type = int,help="delay by step")
    parser.add_argument('--strategy',action = 'store_true',help="if strategy, buffer_size is meaningless")
    parser.add_argument("--hard_update",action='store_true',help="only for step_model")
    parser.add_argument("--random",action='store_true')

    parser.add_argument("--save_interval", default=200, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    args = parser.parse_args()
    main(args)

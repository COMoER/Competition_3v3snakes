import sys
import os
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent)
from agent.agent_eva.common import *
import os

from agent.agent_eva.rnn import RNNQMIX
from agent.agent_eva.mlp import MLPQMIX
import yaml


EPISODE = int(os.environ.get('EPISODE'))

run = int(os.environ.get("RUN"))
QMIX_TYPE = 0
def config():
    dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'rl_trainer_qmix','models','snakes_3v3','run%d'%run)
    print("run%d:"%run)
    f = open(os.path.join(dir,"config.yaml"),'r')
    con = yaml.load(f,Loader=yaml.SafeLoader)
    f.close()
    for query,value in con.items():
        print("%s:  %s"%(query,value))
    print()

config()
try:
    q_agent = MLPQMIX(26,4,3)
    actor_net = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'rl_trainer_qmix', 'models', 'snakes_3v3', 'run%d' % run)
    q_agent.load_model(actor_net, EPISODE)
except:
    q_agent = RNNQMIX(26, 4, 3)
    q_agent.reset()
    actor_net = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'rl_trainer_qmix', 'models', 'snakes_3v3', 'run%d' % run)
    q_agent.load_model(actor_net, EPISODE)
    QMIX_TYPE = 1
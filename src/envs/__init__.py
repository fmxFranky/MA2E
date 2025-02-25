from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from .stag_hunt import StagHunt

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2v2_p"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_z"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_t"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_t"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_t"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_t"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_t_start_1"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_t_start_0"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_t_epo_05"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2v2_t_epo_0"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)

REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")

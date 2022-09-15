print('ciao')


import numpy as np

a = np.array([1., 2.])
print(a)

import os
import shutil
import ray
import ray.rllib.agents.ppo as ppo

import ray.rllib.agents.sac as sac

ray.shutdown()
ray.init(ignore_reinit_error=True)
ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


opz = {"solver": "ppo", # ppo or SAC

}

SELECT_ENV = "CartPole-v0"



CHECKPOINT_ROOT = "tmp/" + SELECT_ENV +"/" + opz['solver']
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
# CHECKPOINT_ROOT = "tmp/ppo/CartPole-v0"


# print("Dashboard URL: http://{}".format(ray.get_webui_url()))


if opz['solver']=="ppo":
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=SELECT_ENV)

elif opz['solver']=="sac":
   config = sac.DEFAULT_CONFIG.copy()
   config["log_level"] = "WARN"
   agent = sac.PPOTrainer(config, env=SELECT_ENV)



# agent.train()

N_ITER = 5
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
  result = agent.train()
  file_name = agent.save(CHECKPOINT_ROOT)

  print(s.format(
    n + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"],
    file_name
   ))
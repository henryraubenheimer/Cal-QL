# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from absl import app, flags

from og_marl.environments import get_environment
from og_marl.loggers import JsonWriter, WandbLogger
from og_marl.offline_dataset import download_and_unzip_vault
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.networks import CNNEmbeddingNetwork
from og_marl.tf2.systems import get_system
from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "smac_v1", "Environment name.")
flags.DEFINE_string("scenario", "3m", "Environment scenario name.")
flags.DEFINE_string("dataset", "Poor", "Dataset type.: 'Good', 'Medium', 'Poor' or 'Replay' ")
flags.DEFINE_string("system", "idrqn+cql", "System name.")
flags.DEFINE_integer("seed", 42, "Seed.")
flags.DEFINE_float("trainer_steps", 50000, "Number of training steps.")
flags.DEFINE_integer("batch_size", 64, "Number of training steps.")


def main(_):
    config = {
        "env": FLAGS.env,
        "scenario": FLAGS.scenario,
        "dataset": FLAGS.dataset,
        "system": FLAGS.system,
        "backend": "tf2",
    }

    env = get_environment(FLAGS.env, FLAGS.scenario)

    buffer = FlashbaxReplayBuffer(sequence_length=20, sample_period=1)

    download_and_unzip_vault(FLAGS.env, FLAGS.scenario)

    is_vault_loaded = buffer.populate_from_vault(FLAGS.env, FLAGS.scenario, FLAGS.dataset, discount=0.99)
    if not is_vault_loaded:
        print("Vault not found. Exiting.")
        return

    logger = WandbLogger(project=FLAGS.system+" - "+FLAGS.scenario)
    #logger = WandbLogger(project=str(FLAGS.trainer_steps)+"_", config=config)

    json_writer = None

    system_kwargs = {
        "add_agent_id_to_obs": True,
        "eps_decay_timesteps": 1, # IMPORTANT: set this to one when doing offline pre-training, else set to 50_000
    }

    system = get_system(FLAGS.system, env, logger, **system_kwargs)

    system.train_offline(buffer, max_trainer_steps=FLAGS.trainer_steps, json_writer=json_writer, evaluate_every=500, num_eval_episodes=4)

    # Swap to online
    system._env_step_ctr = 0.0
    system._cql_weight.assign(0.0)
    system._eps_decay_timesteps = 0

    online_replay_buffer = FlashbaxReplayBuffer(sequence_length=20, sample_period=1)
    system.train_online(online_replay_buffer, max_env_steps=100000, train_period=20)

if __name__ == "__main__":
    app.run(main)

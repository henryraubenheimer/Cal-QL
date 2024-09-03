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

from typing import Any, Dict

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import tree
from chex import Array
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flashbax.vault import Vault

Experience = Dict[str, Array]


class FlashbaxReplayBuffer:
    def __init__(
        self,
        sequence_length: int,
        max_size: int = 50_000,
        batch_size: int = 32,
        sample_period: int = 1,
        seed: int = 42,
    ):
        self._sequence_length = sequence_length
        self._max_size = max_size
        self._batch_size = batch_size

        # Flashbax buffer
        self._replay_buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=batch_size,
            sample_sequence_length=sequence_length,
            period=sample_period,
            min_length_time_axis=1,
            max_size=max_size,
        )

        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
        self._buffer_add_fn = jax.jit(self._replay_buffer.add)

        self._buffer_state: TrajectoryBufferState = None
        self._rng_key = jax.random.PRNGKey(seed)

        self.episode_timesteps = []

    def compute_rewards_to_go(self, experience, discount):
        rewards = experience["rewards"]
        terminals = experience["terminals"]

        B, T, N = rewards.shape

        rewards_to_go = np.zeros((B, T))

        for b in range(B):

            prev_return = 0
            for t in range(T):
                reverse_t = T - t - 1

                rewards_to_go[b, reverse_t] = rewards[b, reverse_t, 0] + discount * prev_return * (1-terminals[b, reverse_t, 0])

                prev_return = rewards_to_go[b, reverse_t]

        rewards_to_go = np.stack([rewards_to_go] * N, axis=2)
        experience["rewards_to_go"] = rewards_to_go

        return experience

    def add(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, np.ndarray],
        terminals: Dict[str, np.ndarray],
        truncations: Dict[str, np.ndarray],
        infos: Dict[str, Any],
        discount: float = 0.99
    ) -> None:
        stacked_infos = {}
        for key, value in infos.items():
            if isinstance(value, dict):
                stacked_infos[key] = np.stack(list(value.values()), axis=0)
            else:
                stacked_infos[key] = value

        timestep = {
            "observations": np.stack(list(observations.values()), axis=0),
            "actions": np.stack(list(actions.values()), axis=0),
            "rewards": np.stack(list(rewards.values()), axis=0),
            "terminals": np.stack(list(terminals.values()), axis=0),
            "truncations": np.stack(list(truncations.values()), axis=0),
            "infos": stacked_infos
        }

        self.episode_timesteps.append(timestep)
        if timestep["terminals"][0] or timestep["truncations"][0]:
            experience = {}
            experience["rewards"] = np.array([np.stack([timestep["rewards"] for timestep in self.episode_timesteps])])
            experience["terminals"] = np.array([np.stack([timestep["terminals"] for timestep in self.episode_timesteps])])
            experience = self.compute_rewards_to_go(experience, discount)

            for i in range(len(self.episode_timesteps)):
                self.episode_timesteps[i]["rewards_to_go"] = experience["rewards_to_go"][0, i, :]

                if self._buffer_state is None:
                    self._buffer_state = self._replay_buffer.init(self.episode_timesteps[i])

                timestep = tree.map_structure(
                    lambda x: jnp.expand_dims(jnp.expand_dims(jnp.array(x), 0), 0), self.episode_timesteps[i]
                )  # add batch & time dims
                self._buffer_state = self._buffer_add_fn(self._buffer_state, timestep)

            self.episode_timesteps = []

        # if self._buffer_state is None:
        #     self._buffer_state = self._replay_buffer.init(timestep)

        # timestep = tree.map_structure(
        #     lambda x: jnp.expand_dims(jnp.expand_dims(jnp.array(x), 0), 0), timestep
        # )  # add batch & time dims
        # self._buffer_state = self._buffer_add_fn(self._buffer_state, timestep)

    def sample(self) -> Experience:
        self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
        batch = self._buffer_sample_fn(self._buffer_state, sample_key)
        return batch.experience  # type: ignore

    def populate_from_vault(
        self, env_name: str, scenario_name: str, dataset_name: str, rel_dir: str = "vaults", discount: float = 0.99
    ) -> bool:
        try:
            self._buffer_state = Vault(
                vault_name=f"{env_name}/{scenario_name}.vlt",
                vault_uid=dataset_name,
                rel_dir=rel_dir,
            ).read()

            # Compute rewards to go
            experience_with_rewards_to_go = self.compute_rewards_to_go(self._buffer_state.experience, discount)

            #make new buffer state
            self._buffer_state = TrajectoryBufferState(
                experience=experience_with_rewards_to_go,
                is_full=self._buffer_state.is_full,
                current_index=self._buffer_state.is_full
            )

            # Recreate the buffer and associated pure functions
            self._replay_buffer = fbx.make_trajectory_buffer(
                add_batch_size=1,
                sample_batch_size=self._batch_size,
                sample_sequence_length=self._sequence_length,
                period=1,
                min_length_time_axis=1,
                max_size=self._sequence_length,
            )
            self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
            self._buffer_add_fn = jax.jit(self._replay_buffer.add)

            return True

        except ValueError:
            return False

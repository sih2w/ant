from __future__ import annotations
import argparse
import glob
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import scavenging_ant.envs.scavenging_ant as scavenging_ant
from torch import optim
from tqdm import tqdm

class A2C(nn.Module):
    def __init__(
            self,
            feature_count: int,
            action_count: int,
            device: torch.device,
            critic_learning_rate: float = 1e-5,
            actor_learning_rate: float = 1e-3,
            environment_count: int = 1,
            hidden_count: int = 100,
    ):
        super().__init__()
        self.__device = device
        self.__environment_count = environment_count

        critic_layers = [
            nn.Linear(feature_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, 1),
        ]

        actor_layers = [
            nn.Linear(feature_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, action_count),
        ]

        self.__critic = nn.Sequential(*critic_layers).to(self.__device)
        self.__actor = nn.Sequential(*actor_layers).to(self.__device)
        self.__critic_optimizer = optim.RMSprop(self.__critic.parameters(), lr=critic_learning_rate)
        self.__actor_optimizer = optim.RMSprop(self.__actor.parameters(), lr=actor_learning_rate)

    def get_actor(self):
        return self.__actor

    def get_critic(self):
        return self.__critic

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(x).to(self.__device)
        state_values = self.__critic(x)
        action_logit_vec = self.__actor(x)
        return state_values, action_logit_vec

    def select_action(
            self,
            x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_values, action_logit = self.forward(x)
        action_pd = torch.distributions.Categorical(logits=action_logit)
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_values, entropy

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probabilities: torch.Tensor,
        value_predictions: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        entropy_coefficient: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reward_length = len(rewards)
        advantages = torch.zeros(reward_length, self.__environment_count, device=device)
        gae = 0.0

        for t in reversed(range(reward_length - 1)):
            td_error = (
                    rewards[t] + gamma * masks[t] * value_predictions[t + 1] - value_predictions[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        critic_loss = advantages.pow(2).mean()
        actor_loss = (
                -(advantages.detach() * action_log_probabilities).mean() - entropy_coefficient * entropy.mean()
        )
        return critic_loss, actor_loss

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        self.__critic_optimizer.zero_grad()
        critic_loss.backward()
        self.__critic_optimizer.step()

        self.__actor_optimizer.zero_grad()
        actor_loss.backward()
        self.__actor_optimizer.step()

def get_default_domain_environment(args, **kwargs):
    environment = scavenging_ant.ScavengingAntEnv(
        persistent_obstacles=True,
        persistent_food=True,
        persistent_nests=True,
        nest_count=args.nest_count,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        food_count=args.food_count,
        percent_obstacles=args.percent_obstacles,
        seed=args.seed,
        **kwargs,
    )
    return environment

def get_randomized_domain_environment(args, **kwargs):
    environment = scavenging_ant.ScavengingAntEnv(
        persistent_obstacles=True,
        persistent_food=True,
        persistent_nests=True,
        nest_count=args.nest_count,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        food_count=args.food_count,
        percent_obstacles=np.random.uniform(0, 0.10),
        seed=args.seed,
        **kwargs,
    )
    return environment

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nest-count", type=int, default=1)
    parser.add_argument("--grid-height", type=int, default=10)
    parser.add_argument("--grid-width", type=int, default=18)
    parser.add_argument("--food-count", type=int, default=1)
    parser.add_argument("--percent-obstacles", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment-count", type=int, default=10)
    parser.add_argument("--critic-learning-rate", type=float, default=0.005)
    parser.add_argument("--actor-learning-rate", type=float, default=0.001)
    parser.add_argument("--update-count", type=int, default=1000)
    parser.add_argument("--steps-per-update", type=int, default=108)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--lam", type=float, default=0.99)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--use-cuda", type=bool, default=False)
    parser.add_argument("--save-weights", type=bool, default=True)
    parser.add_argument("--showcase-episode-count", type=int, default=10)
    parser.add_argument("--plot-results", type=bool, default=True)
    parser.add_argument("--randomize-domain", type=bool, default=False)
    parser.add_argument("--watch-recent", type=bool, default=False)
    return parser

def plot_results(environment, entropies, critic_losses, actor_losses, args):
    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    fig.suptitle(
        f"Training plots for {agent.__class__.__name__}\nenvironment_count={args.environment_count}, steps_per_update={args.steps_per_update}"
    )

    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
            np.convolve(
                np.array(environment.return_queue).flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
    )
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / args.environment_count,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
            np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")

    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
            np.convolve(
                np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")

    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
            np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    plt.show()

def train(environment, args):
    critic_losses = []
    actor_losses = []
    entropies = []

    for sample_phase in tqdm(range(args.update_count)):
        episode_value_predictions = torch.zeros(args.steps_per_update, args.environment_count, device=device)
        ep_rewards = torch.zeros(args.steps_per_update, args.environment_count, device=device)
        ep_action_log_probs = torch.zeros(args.steps_per_update, args.environment_count, device=device)
        masks = torch.zeros(args.steps_per_update, args.environment_count, device=device)

        if sample_phase == 0:
            states, info = environment.reset(seed=42)

        for step in range(args.steps_per_update):
            actions, action_log_probs, state_value_predictions, entropy = agent.select_action(states)
            states, rewards, terminated, truncated, infos = environment.step(actions.cpu().numpy())

            episode_value_predictions[step] = torch.squeeze(state_value_predictions)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs

            masks[step] = torch.tensor([not term for term in terminated])

        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            episode_value_predictions,
            entropy,
            masks,
            args.gamma,
            args.lam,
            args.entropy_coefficient,
            device,
        )

        agent.update_parameters(critic_loss, actor_loss)

        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())

    return critic_losses, actor_losses, entropies

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    base_directory = "a2c_weights"
    timestamp_format = "%Y%m%d-%H%M%S"

    if not os.path.exists(base_directory):
        os.mkdir(base_directory)

    if args.randomize_domain:
        environment = gym.vector.AsyncVectorEnv([
            lambda: get_randomized_domain_environment(args) for _ in range(args.environment_count)
        ])
    else:
        environment = gym.vector.AsyncVectorEnv([
            lambda: get_default_domain_environment(args) for _ in range(args.environment_count)
        ])

    environment = gym.wrappers.vector.FlattenObservation(environment)
    environment = gym.wrappers.vector.NormalizeReward(environment)
    observation_shape = environment.single_observation_space.shape[0]
    action_shape = environment.single_action_space.n

    print("Observation shape:", observation_shape)
    print("Action shape:", action_shape)

    if args.use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    agent = A2C(
        observation_shape,
        action_shape,
        device,
        args.critic_learning_rate,
        args.actor_learning_rate,
        args.environment_count,
    )

    if args.watch_recent:
        environment.close()

        current_timestamp = time.strftime(timestamp_format)
        pattern = os.path.join(base_directory, "*")

        subfolders = [folder for folder in glob.glob(pattern) if os.path.isdir(folder)]
        matching_folders = [folder for folder in subfolders if os.path.basename(folder).count("-") == 1]
        matching_folders.sort()

        if matching_folders:
            most_recent_folder = matching_folders[-1]
            agent.get_actor().load_state_dict(torch.load(f"{most_recent_folder}\\actor_weights.h5"))
            agent.get_critic().load_state_dict(torch.load(f"{most_recent_folder}\\critic_weights.h5"))
            agent.get_actor().eval()
            agent.get_critic().eval()

            if args.randomize_domain:
                environment = get_randomized_domain_environment(args, render_mode="human", render_fps=10)
            else:
                environment = get_default_domain_environment(args, render_mode="human", render_fps=10)
            environment = gym.wrappers.FlattenObservation(environment)

            for episode in range(args.showcase_episode_count):
                state, info = environment.reset()
                done = False
                while not done:
                    with torch.no_grad():
                        action, _, _, _ = agent.select_action(state[None, :])
                    state, reward, terminated, truncated, info = environment.step(action.item())
                    done = terminated or truncated

            environment.close()
        else:
            print("Existing weights do not exist.")
            exit(0)
    else:
        environment = gym.wrappers.vector.RecordEpisodeStatistics(
            environment,
            buffer_length=args.environment_count * args.update_count
        )

        critic_losses, actor_losses, entropies = train(environment, args)
        if args.plot_results:
            plot_results(environment, entropies, critic_losses, actor_losses, args)
        environment.close()

        if args.save_weights:
            base_directory = f"{base_directory}/{time.strftime(timestamp_format)}"
            if not os.path.exists(base_directory):
                os.mkdir(base_directory)

            torch.save(agent.get_actor().state_dict(), f"{base_directory}/actor_weights.h5")
            torch.save(agent.get_critic().state_dict(), f"{base_directory}/critic_weights.h5")
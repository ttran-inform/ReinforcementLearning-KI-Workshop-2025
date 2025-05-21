import numpy as np
import gymnasium as gym
import random
from tqdm import tqdm
import imageio
import os

# Create the Blackjack environment
# natural=False means that getting a natural blackjack doesn't give extra reward
# sab=False disables the simplified version of the action space
env = gym.make('Blackjack-v1', render_mode="rgb_array",
               natural=False, sab=False)

# Print info about observation and action spaces
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space.n} possible actions")


def initialize_q_table(observation_space, action_space):
    """
    Initialize Q-table with zeros
    For Blackjack, the state is a tuple (player_sum, dealer_card, usable_ace)
    """
    # Players sum: 0-31, Dealer showing: 1-10, Usable ace: 0-1
    Qtable = np.zeros((32, 11, 2, action_space))
    return Qtable


# Initialize Q-table
Qtable_blackjack = initialize_q_table(
    env.observation_space, env.action_space.n)
print("Q-table shape:", Qtable_blackjack.shape)


def greedy_policy(Qtable, state):
    """
    Greedy policy - choose action with highest Q-value
    """
    player_sum, dealer_card, usable_ace = state
    # Take the action with the highest expected future reward
    action = np.argmax(Qtable[player_sum, dealer_card, usable_ace, :])
    return action


def epsilon_greedy_policy(Qtable, state, epsilon):
    """
    Epsilon-greedy policy - explore with probability epsilon, exploit otherwise
    """
    random_num = random.uniform(0, 1)
    # If random number is greater than epsilon, exploit (greedy action)
    if random_num > epsilon:
        action = greedy_policy(Qtable, state)
    # Otherwise explore (random action)
    else:
        action = env.action_space.sample()
    return action


# Training parameters
# Total training episodes
n_training_episodes = 5000
learning_rate = 0.1           # Learning rate
max_steps = 100               # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability
# Exponential decay rate for exploration prob (slower for Blackjack)
decay_rate = 0.00005

# Evaluation parameters
# Total number of test episodes (more for better evaluation)
n_eval_episodes = 1000
eval_seed = 123               # Seed for evaluation


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    """
    Train the Q-learning agent
    """
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-decay_rate * episode)

        # Reset the environment
        state, info = env.reset()
        terminated = False
        truncated = False

        # Loop for max_steps
        for _ in range(max_steps):
            # Choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take action and observe next state and reward
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q-table using the Q-learning update rule
            player_sum, dealer_card, usable_ace = state
            new_player_sum, new_dealer_card, new_usable_ace = new_state

            # Q(s,a) := Q(s,a) + lr * [R + gamma * max Q(s',a') - Q(s,a)]
            Qtable[player_sum, dealer_card, usable_ace, action] = Qtable[player_sum, dealer_card, usable_ace, action] + \
                learning_rate * (reward + gamma * np.max(Qtable[new_player_sum, new_dealer_card, new_usable_ace, :]) -
                                 Qtable[player_sum, dealer_card, usable_ace, action])

            # Break if game ended
            if terminated or truncated:
                break

            # Update state
            state = new_state

    return Qtable


# Train the agent
print("Training the agent...")
Qtable_blackjack = train(n_training_episodes, min_epsilon,
                         max_epsilon, decay_rate, env, max_steps, Qtable_blackjack)
print("Training complete!")


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed=None):
    """
    Evaluate the agent
    """
    episode_rewards = []
    wins = 0
    draws = 0
    losses = 0

    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        total_rewards_ep = 0
        terminated = False
        truncated = False

        for _ in range(max_steps):
            # Take greedy action
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                # Count wins, draws, and losses
                if reward > 0:
                    wins += 1
                elif reward == 0:
                    draws += 1
                else:
                    losses += 1
                break

            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    win_rate = wins / n_eval_episodes
    draw_rate = draws / n_eval_episodes
    loss_rate = losses / n_eval_episodes

    return mean_reward, std_reward, win_rate, draw_rate, loss_rate


# Evaluate the agent
print("Evaluating agent...")
mean_reward, std_reward, win_rate, draw_rate, loss_rate = evaluate_agent(
    env, max_steps, n_eval_episodes, Qtable_blackjack, eval_seed)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(
    f"Win rate: {win_rate:.2%}, Draw rate: {draw_rate:.2%}, Loss rate: {loss_rate:.2%}")


def record_video(env, Qtable, out_directory, fps=1, num_episodes=3):
    """
    Generate replay videos of the agent
    """
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(out_directory)):
        os.makedirs(os.path.dirname(out_directory))

    for episode in range(num_episodes):
        images = []
        state, info = env.reset(seed=random.randint(0, 500))
        img = env.render()
        images.append(img)

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Take greedy action
            player_sum, dealer_card, usable_ace = state
            action = np.argmax(Qtable[player_sum, dealer_card, usable_ace, :])

            # Get next state
            state, reward, terminated, truncated, info = env.step(action)

            # Render and save image
            img = env.render()
            images.append(img)

        # Save video
        episode_path = out_directory.replace('.gif', f'_episode_{episode}.gif')
        imageio.mimsave(episode_path, [np.array(img)
                        for img in images], fps=fps)
        print(f"Episode {episode} saved to {episode_path}")


# Record videos of agent playing
video_directory = "Blackjack/videos/blackjack.gif"
record_video(env, Qtable_blackjack, video_directory, fps=2, num_episodes=5)

env.close()

# Print a basic strategy table based on our learned Q-function
print("\nLearned strategy (player sum, dealer showing, usable ace):")
print("Player sum | Dealer card | Usable ace | Action (0: Stick, 1: Hit)")
for player_sum in range(11, 22):  # Only display from 11 to 21
    for dealer_card in range(1, 11):
        for usable_ace in [0, 1]:
            action = np.argmax(
                Qtable_blackjack[player_sum, dealer_card, usable_ace, :])
            action_name = "Stick" if action == 0 else "Hit"
            print(
                f"{player_sum:10} | {dealer_card:11} | {usable_ace:10} | {action_name}")

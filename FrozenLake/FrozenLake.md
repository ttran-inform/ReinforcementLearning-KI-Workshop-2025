# 🧠 Q-Learning Parameters Explained (for FrozenLake)

| Parameter             | What It Does (Simple)                                             | ↑ Increase →                                               | ↓ Decrease →                                                  |
| --------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- |
| `learning_rate` (`α`) | Controls how fast the agent updates its Q-values after each step. | Learns faster, but may overshoot or become unstable.       | Learns slower, more stable but may converge slowly.           |
| `gamma` (`γ`)         | Controls how much the agent cares about **future rewards**.       | Thinks more long-term; plans ahead.                        | Focuses only on short-term rewards.                           |
| `n_training_episodes` | How many games the agent plays to learn.                          | More training, better final performance (takes longer).    | Less training, faster but possibly worse performance.         |
| `max_steps`           | Maximum steps allowed in one episode.                             | Lets agent explore more per game.                          | Less exploration; episodes may end before learning.           |
| `max_epsilon`         | Starting chance of **exploring** (random moves).                  | Explores more at the beginning (more diverse experiences). | Explores less; may miss good paths early on.                  |
| `min_epsilon`         | Minimum chance of exploration by the end of training.             | Keeps exploring longer, may delay learning.                | Agent becomes greedy sooner; learns faster but may get stuck. |
| `decay_rate`          | How fast exploration (`epsilon`) drops from max to min over time. | Agent becomes greedy faster (less exploration).            | Explores longer, learns cautiously.                           |

High learning rate + low exploration = fast learning, but risky.

Low learning rate + high exploration = slow but thorough learning.

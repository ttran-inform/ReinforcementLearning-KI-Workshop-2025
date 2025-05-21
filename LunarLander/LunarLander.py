from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym  # Importiere Gymnasium für die Umgebung
from gymnasium.wrappers import RecordVideo  # Wrapper zum Aufzeichnen von Videos

env = make_vec_env('LunarLander-v2', n_envs=16)

# Initialisiere das PPO-Modell mit den gewünschten Hyperparametern
# PPO ist eine Kombination aus: Value-based reinforcement learning & Policy-based reinforcement learning method

model = PPO(
    policy='MlpPolicy',    # Wähle eine mehrschichtige Wahrnehmungspolicy (MLP)
    env=env,               # Übergib die Vektor-Umgebung
    n_steps=1024,          # Anzahl der Schritte pro Rollout
    batch_size=64,         # Batch-Größe für das Training
    n_epochs=4,            # Anzahl der Epochen pro Update
    gamma=0.999,           # Diskontierungsfaktor
    gae_lambda=0.98,       # GAE-Lambda für Generalized Advantage Estimation
    ent_coef=0.01,         # Entropie-Koeffizient zur Förderung der Exploration
    verbose=1              # Ausführliches Logging aktivieren
)

# Starte das Training Zeitschritte
model.learn(total_timesteps=1)

# Name, unter dem das Modell gespeichert werden könnte
model_name = "ppo-LunarLander-v2"
# model.save(model_name)  # Zum Speichern des Modells entkommentieren.

# Erstelle eine Einzelumgebung für die Evaluation mit Video-Aufzeichnung
eval_env = gym.make("LunarLander-v2", render_mode='rgb_array')
eval_env = RecordVideo(
    env=eval_env,  # Verzeichnis zum Speichern der Videos
    video_folder="LunarLander/videos",  # Zeichne die ersten 5 Episoden auf
    episode_trigger=lambda episode_id: episode_id < 5,
    name_prefix="ppo_eval"  # Präfix für die Videodateinamen
)
# Füge Monitor hinzu, um Belohnungen und andere Statistiken zu protokollieren
eval_env = Monitor(eval_env)

# Bewerte die gelernte Policy über 5 deterministische Episoden
mean_reward, std_reward = evaluate_policy(
    model, eval_env,
    n_eval_episodes=5,
    deterministic=True
)

# Gib den Mittelwert und die Standardabweichung der Belohnungen aus
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Schließe die Evaluationsumgebung und speichere alle Videos
eval_env.close()

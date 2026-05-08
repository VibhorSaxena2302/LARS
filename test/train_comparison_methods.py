import os
import pickle
import random
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# GLOBAL CONFIG
# ============================================================
ENV_NAME = "MountainCar-v0"
SEED = 42
MAX_STEPS = 1000

# ---- Training episodes for all deep methods ----
DQN_EPISODES = 500

# ---- DQN Hyperparams (keep SAME for fairness) ----
GAMMA_RL = 0.99
LR_AGENT = 5e-4
REPLAY_CAPACITY = 50000
BATCH_SIZE = 64
EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.995
TAU_SOFT = 0.01

# ---- Shaping hyperparams ----
GAMMA_SHAPING = GAMMA_RL
ALPHA_START = 2.0
ALPHA_DECAY = 0.95
SHAPING_UPDATE_FREQ = 10  # episodes

# ---- Potential network hyperparams ----
POT_HIDDEN1 = 64
POT_HIDDEN2 = 32
LR_POT = 1e-3
POT_PRETRAIN_EPOCHS = 50
POT_UPDATE_EPOCHS = 10

# ---- Demo generation (from tabular Q) ----
NUM_DEMO_EPISODES = 5
DEMO_GAMMA = GAMMA_RL

# ---- Q-learning baseline discretization ----
POS_SEGMENTS = 20
VEL_SEGMENTS = 20
Q_LR = 0.9
Q_GAMMA = 0.9
Q_EPISODES = DQN_EPISODES

SHAPING_CLIP = 1.0
SHAPING_CLIP_MIN = 0.2
SHAPING_CLIP_DECAY = 0.995 

# LARS-specific
LARS_UPDATE_FREQ = 10
LARS_ALPHA_DECAY = 0.80
LARS_ALPHA_MIN = 0.05
LARS_INIT_DEMOS = 25      # collect more initial demos
LARS_INIT_TOPK = 8
LARS_UPDATE_DEMOS = 10    # per update
LARS_UPDATE_TOPK = 3
LARS_STAGE1_TARGET = 200   # get under 200 fast
LARS_STAGE2_TARGET = 150   # then push under 150
LARS_ALPHA_DECAY_STAGE1 = 0.98   # decay slowly while still >200
LARS_ALPHA_DECAY_STAGE2 = 0.90   # decay faster after you’re <=200
LARS_BOOST_EPISODES = 80
LARS_ALPHA_BOOST = 1.25   # mild boost; don’t go crazy

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# UTIL: make sure directory exists
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# TABULAR Q BASELINE (for demos + baseline comparison)
# ============================================================
def train_tabular_q(save_dir="models/q", version="v1"):
    ensure_dir(save_dir)
    env = gym.make(ENV_NAME, max_episode_steps=MAX_STEPS, render_mode=None)
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], POS_SEGMENTS)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], VEL_SEGMENTS)
    q_table = np.zeros((POS_SEGMENTS + 1, VEL_SEGMENTS + 1, env.action_space.n), dtype=np.float32)

    eps = 1.0
    eps_decay_rate = 2.0 / Q_EPISODES
    eps_min = 0

    true_rewards = []

    for ep in range(Q_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            sp = np.digitize(obs[0], pos_space)
            sv = np.digitize(obs[1], vel_space)

            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[sp, sv]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            nsp = np.digitize(next_obs[0], pos_space)
            nsv = np.digitize(next_obs[1], vel_space)

            best_next = np.max(q_table[nsp, nsv])
            q_table[sp, sv, action] += Q_LR * (reward + Q_GAMMA * best_next - q_table[sp, sv, action])

            total_reward += reward
            obs = next_obs
            steps += 1

        eps = max(eps - eps_decay_rate, eps_min)
        true_rewards.append(total_reward)
        print(f"[Q] Episode {ep+1} TrueReward: {total_reward:.2f} Eps:{eps:.2f}")

    # Save Q agent
    out_path = os.path.join(save_dir, f"mountain_car_{version}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((q_table, pos_space, vel_space), f)

    # Plot (smoothed)
    window = 100
    mean_rewards = np.zeros(len(true_rewards))
    for t in range(len(true_rewards)):
        mean_rewards[t] = np.mean(true_rewards[max(0, t-window):(t+1)])

    plt.figure(figsize=(8, 4))
    plt.plot(mean_rewards, label=f"Moving Avg ({window}) - True Env Reward (Q-learning)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Tabular Q-learning on MountainCar (smoothed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"q_training_{version}.png"))
    plt.close()

    np.save(os.path.join(save_dir, f"q_rewards_{version}.npy"), np.array(true_rewards, dtype=np.float32))
    env.close()
    return out_path


def load_discrete_q_agent(path):
    with open(path, "rb") as f:
        q_table, pos_space, vel_space = pickle.load(f)
    return q_table, pos_space, vel_space


def collect_demonstrations_best(
    q_table,
    pos_space,
    vel_space,
    num_eps=NUM_DEMO_EPISODES,
    top_k=3,
    gamma=DEMO_GAMMA,
    seed_offset=1000
):
    """
    Returns demo_states (N, state_dim), demo_returns (N,)
    Keeps the best episodes primarily by fewer steps, tie-break by higher total reward.
    """
    env = gym.make(ENV_NAME, max_episode_steps=MAX_STEPS, render_mode=None)
    episodes = []

    for ep in range(num_eps):
        obs, _ = env.reset(seed=SEED + seed_offset + ep)
        traj = []
        done = False
        steps = 0
        total_r = 0.0

        while not done and steps < MAX_STEPS:
            sp = np.digitize(obs[0], pos_space)
            sv = np.digitize(obs[1], vel_space)
            action = int(np.argmax(q_table[sp, sv]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            traj.append((obs, action, reward, next_obs, done))
            total_r += reward
            obs = next_obs
            steps += 1

        episodes.append({"traj": traj, "total_r": total_r, "steps": steps})

    env.close()

    top_k = min(top_k, len(episodes))
    episodes.sort(key=lambda e: (e["steps"], -e["total_r"]))
    best_eps = episodes[:top_k]

    demo_states, demo_returns = [], []
    for e in best_eps:
        traj = e["traj"]
        demo_states, demo_returns, demo_transitions = [], [], []
        T = len(traj)
        for t, (s, a, r, s2, done) in enumerate(traj):
            remaining = T - t
            demo_states.append(s)
            demo_returns.append(-float(remaining))
            demo_transitions.append((s, a, r, s2, float(done)))



    demo_states = np.array(demo_states, dtype=np.float32)
    demo_returns = np.array(demo_returns, dtype=np.float32)

    print(f"Collected {num_eps} eps, kept top_k={top_k}, samples={len(demo_states)}")
    print("Best eps (steps, total_reward):", [(e["steps"], e["total_r"]) for e in best_eps])

    return demo_states, demo_returns, demo_transitions


# ============================================================
# DQN
# ============================================================
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.qnet = QNet(state_dim, action_dim).to(device)
        self.target = QNet(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.opt = optim.Adam(self.qnet.parameters(), lr=LR_AGENT)

        self.gamma = GAMMA_RL
        self.eps = EPS_START
        self.eps_min = EPS_MIN
        self.eps_decay = EPS_DECAY

        self.replay = deque(maxlen=REPLAY_CAPACITY)
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.qnet(s)
        return int(q.argmax(dim=1).item())

    def store(self, s, a, r, s2, done):
        self.replay.append((s, a, r, s2, done))

    def update(self):
        if len(self.replay) < BATCH_SIZE:
            return

        batch = random.sample(self.replay, BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        qsa = self.qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next = self.target(s2).max(dim=1)[0]
            y = r + self.gamma * max_next * (1 - d)

        loss = nn.functional.mse_loss(qsa, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.opt.step()

    def soft_update(self, tau=TAU_SOFT):
        for p, tp in zip(self.qnet.parameters(), self.target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ============================================================
# Potential Networks (MLP and RNN)
# ============================================================
class PotentialNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, POT_HIDDEN1),
            nn.LayerNorm(POT_HIDDEN1),
            nn.LeakyReLU(0.01),
            nn.Linear(POT_HIDDEN1, POT_HIDDEN2),
            nn.LayerNorm(POT_HIDDEN2),
            nn.LeakyReLU(0.01),
            nn.Linear(POT_HIDDEN2, 1)
        )

    def forward(self, x):
        return self.net(x)


class RNNPotentialNet(nn.Module):
    """
    Lightweight GRU-based potential: Φ(s_t, h_t)
    """
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRUCell(state_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_dim, device=device)

    def forward(self, x, h):
        h2 = self.gru(x, h)
        phi = self.head(h2)
        return phi, h2


# ============================================================
# Shaping Wrapper (PBRS)
# r_total = r_env + alpha*(gamma*Phi(s') - Phi(s))
# ============================================================
class ShapingWrapper(gym.Wrapper):
    def __init__(self, env, phi_model, device, gamma=GAMMA_SHAPING, alpha=ALPHA_START, rnn=False):
        super().__init__(env)
        self.phi = phi_model
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.rnn = rnn
        self.clip = SHAPING_CLIP

        self.prev_phi = None
        self.h = None  # for RNN mode

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]

        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.rnn:
                self.h = self.phi.init_hidden(self.device)
                phi_val, self.h = self.phi(x, self.h)
                self.prev_phi = float(phi_val.item())
            else:
                self.prev_phi = float(self.phi(x).item())
        return obs, info

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]

        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if terminated or truncated:
                phi_next = 0.0
            else:
                if self.rnn:
                    phi_val, self.h = self.phi(x, self.h)
                    phi_next = float(phi_val.item())
                else:
                    phi_next = float(self.phi(x).item())

        r_shaping = self.gamma * phi_next - self.prev_phi
        r_shaping = float(np.clip(r_shaping, -self.clip, self.clip)) 
        r_total = r_env + self.alpha * r_shaping
        self.prev_phi = phi_next

        info = dict(info)
        info["r_env"] = float(r_env)
        info["r_shaping"] = float(self.alpha * r_shaping)

        return obs, r_total, terminated, truncated, info


# ============================================================
# Evaluation on TRUE env reward (no shaping)
# ============================================================
def evaluate_policy_once(env, agent, device, seed=None):
    obs, _ = env.reset(seed=seed)
    if isinstance(obs, tuple):
        obs = obs[0]

    total = 0.0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = agent.qnet(x)
        action = int(q.argmax(dim=1).item())

        obs2, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if isinstance(obs2, tuple):
            obs2 = obs2[0]

        total += r
        obs = obs2
        steps += 1

    return total


# ============================================================
# Train potential on (states -> returns)
# ============================================================
def train_potential_mlp(phi_net, opt, states_np, returns_np, device, epochs):
    """
    Trains phi(s) to predict normalized returns (z-score).
    Using smooth_l1_loss (Huber) to reduce sensitivity to outliers.
    """
    phi_net.train()
    states = torch.tensor(states_np, dtype=torch.float32, device=device)

    y = np.asarray(returns_np, dtype=np.float32).reshape(-1, 1)
    y = y / float(MAX_STEPS)   # now roughly in [-1, 0]
    targets = torch.tensor(y, dtype=torch.float32, device=device)

    bs = 64  # slightly larger helps stabilize phi training
    for _ in range(epochs):
        perm = np.random.permutation(len(states))
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            s_b = states[idx]
            t_b = targets[idx]
            opt.zero_grad()
            pred = phi_net(s_b)
            loss = nn.functional.smooth_l1_loss(pred, t_b)  # Huber
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phi_net.parameters(), 5.0)
            opt.step()

    phi_net.eval()


def mean_last(x, k):
    if len(x) < k:
        return float(np.mean(x)) if len(x) else 0.0
    return float(np.mean(x[-k:]))

def evaluate_policy_avg(agent, device, seeds, max_steps=MAX_STEPS):
    env = gym.make(ENV_NAME, max_episode_steps=max_steps, render_mode=None)
    totals = []
    for s in seeds:
        totals.append(evaluate_policy_once(env, agent, device=device, seed=s))
    env.close()
    return float(np.mean(totals))

# ============================================================
# Main trainer (method selector)
# Methods supported:
#   - lars
#   - demo_static
#   - brs
#   - dpbrs
#   - rnn
# ============================================================
def train_method(method="lars", version="v1", q_demo_path="models/q/mountain_car_v1.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Prepare env
    env = gym.make(ENV_NAME, max_episode_steps=MAX_STEPS, render_mode=None)
    obs, _ = env.reset(seed=SEED)
    if isinstance(obs, tuple):
        obs = obs[0]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Directories
    save_dir = os.path.join("models", method)
    ensure_dir(save_dir)

    # Agent
    agent = DQNAgent(state_dim, action_dim, device=device)

    # Rewards logs
    shaped_rewards = []
    true_rewards = []

    lars_frozen = False
    LARS_FREEZE_THRESHOLD = 150   # ~150 steps
    LARS_FREEZE_PATIENCE = 20      # last-20 avg

    DEMO_FREEZE_THRESHOLD = -160   # (~160 steps)
    DEMO_FREEZE_PATIENCE = 20
    DEMO_ALPHA_DECAY = 0.90
    DEMO_ALPHA_MIN = 0.00
    demo_frozen = False

    # ========================================================
    # Setup shaping potential depending on method
    # ========================================================
    alpha = ALPHA_START

    phi_net = None
    phi_opt = None
    rnn_mode = False

    # ------ Methods that need MLP potential ------
    if method in ["lars", "demo_static", "brs", "dpbrs"]:
        phi_net = PotentialNet(state_dim).to(device)
        phi_opt = optim.Adam(phi_net.parameters(), lr=LR_POT)

    # ------ RNN potential ------
    if method == "rnn":
        phi_net = RNNPotentialNet(state_dim, hidden_dim=32).to(device)
        phi_opt = optim.Adam(phi_net.parameters(), lr=LR_POT)
        rnn_mode = True

    # ------ Demo-based init (LARS/Demo/BRS) ------
    demo_states = None
    demo_returns = None
    if method in ["lars", "demo_static", "brs"]:
        if not os.path.exists(q_demo_path):
            raise FileNotFoundError(
                f"Demo Q model not found at {q_demo_path}. Train tabular Q first."
            )
        q_table, pos_space, vel_space = load_discrete_q_agent(q_demo_path)
        demo_states, demo_returns, demo_transitions = collect_demonstrations_best(
            q_table, pos_space, vel_space,
            num_eps=LARS_INIT_DEMOS,
            top_k=LARS_INIT_TOPK,
            gamma=DEMO_GAMMA
        )
        train_potential_mlp(phi_net, phi_opt, demo_states, demo_returns, device, POT_PRETRAIN_EPOCHS)

    # ------ DPBRS init: near-zero potential (optional but good) ------
    if method == "dpbrs":
        for m in phi_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    # Create shaped env wrapper for shaping methods; for Q baseline this isn't used.
    shaped_env = ShapingWrapper(env, phi_net, device=device, gamma=GAMMA_SHAPING, alpha=alpha, rnn=rnn_mode)

    # DPBRS / RNN need buffers to train potential from agent experience
    interval_states = []
    interval_rewards = []

    if method == "lars":
        # Prefill replay so DQN learns immediately from expert-like transitions
        for (s, a, r, s2, d) in demo_transitions:
            # IMPORTANT: store ENV reward here, not shaped (stable)
            agent.store(s, a, r, s2, d)

        # Optional: brief warmup updates so policy improves before exploration
        for _ in range(500):
            agent.update()
            agent.soft_update()

    # ========================================================
    # TRAIN LOOP
    # ========================================================
    for ep in range(DQN_EPISODES):
        obs, _ = shaped_env.reset(seed=SEED + ep)
        ep_shaped = 0.0
        done = False
        steps = 0

        # store full episode for DPBRS/RNN potential updates
        ep_states = []
        ep_env_rewards = []

        while not done and steps < MAX_STEPS:
            action = agent.select_action(obs)
            next_obs, r_total, terminated, truncated, info = shaped_env.step(action)
            done = terminated or truncated

            # Store transitions for DQN (uses shaped reward)
            agent.store(obs, action, r_total, next_obs, float(done))
            agent.update()
            agent.soft_update()

            ep_shaped += r_total

            # For DPBRS / RNN potential learning, store true env reward too:
            # env reward in MountainCar = -1 per step, but we can reconstruct:
            # r_env = -1 always, so:
            r_env = info.get("r_env", -1.0)
            ep_states.append(obs)
            ep_env_rewards.append(r_env)

            obs = next_obs
            steps += 1

        # end episode
        agent.decay_eps()
        shaped_rewards.append(ep_shaped)

        # Evaluate true reward
        eval_seeds = [SEED, SEED+1, SEED+2, SEED+3, SEED+4]
        true_r = evaluate_policy_avg(agent, device=device, seeds=eval_seeds)

        true_rewards.append(true_r)

        if method == "demo_static" and (not demo_frozen) and ep >= 50:
            recent = mean_last(true_rewards, DEMO_FREEZE_PATIENCE)
            if recent > DEMO_FREEZE_THRESHOLD:
                demo_frozen = True
                shaped_env.alpha = 0.0

        if method == "lars" and (not lars_frozen):
            if ep < LARS_BOOST_EPISODES:
                shaped_env.alpha = max(shaped_env.alpha, LARS_ALPHA_BOOST)

        if method == "lars":
            shaped_env.clip = max(SHAPING_CLIP_MIN, shaped_env.clip * SHAPING_CLIP_DECAY)

        if method == "lars" and (not lars_frozen) and ep >= 50:
            recent_steps = -mean_last(true_rewards, LARS_FREEZE_PATIENCE)
            if recent_steps <= LARS_FREEZE_THRESHOLD:
                lars_frozen = True

        if method == "lars" and lars_frozen:
            shaped_env.alpha *= 0.5
            if shaped_env.alpha < 0.02:
                shaped_env.alpha = 0.0

        # Collect for DPBRS/RNN interval training
        if method in ["dpbrs", "rnn"]:
            interval_states.append(ep_states)
            interval_rewards.append(ep_env_rewards)

        # ====================================================
        # POTENTIAL UPDATES / ALPHA DECAYS
        # ====================================================
        if (ep + 1) % SHAPING_UPDATE_FREQ == 0:
            # LARS: update potential using more demo samples + decay alpha
            if method == "lars":
                if not lars_frozen:
                    q_table, pos_space, vel_space = load_discrete_q_agent(q_demo_path)

                    new_s, new_ret, _ = collect_demonstrations_best(
                        q_table, pos_space, vel_space,
                        num_eps=LARS_UPDATE_DEMOS,
                        top_k=LARS_UPDATE_TOPK,
                        gamma=DEMO_GAMMA,
                        seed_offset=5000 + ep
                    )

                    MAX_DEMO_SAMPLES = 8000
                    demo_states = np.concatenate([demo_states, new_s], axis=0)
                    demo_returns = np.concatenate([demo_returns, new_ret], axis=0)
                    if len(demo_states) > MAX_DEMO_SAMPLES:
                        demo_states = demo_states[-MAX_DEMO_SAMPLES:]
                        demo_returns = demo_returns[-MAX_DEMO_SAMPLES:]

                    train_potential_mlp(phi_net, phi_opt, demo_states, demo_returns, device, POT_UPDATE_EPOCHS)

                    recent_steps = -mean_last(true_rewards, 20)

                    if recent_steps > LARS_STAGE1_TARGET:
                        shaped_env.alpha = max(LARS_ALPHA_MIN, shaped_env.alpha * LARS_ALPHA_DECAY_STAGE1)
                    else:
                        shaped_env.alpha = max(LARS_ALPHA_MIN, shaped_env.alpha * LARS_ALPHA_DECAY_STAGE2)


            # Demo_static: no updates, no decay
            elif method == "demo_static":
                if demo_frozen:
                    shaped_env.alpha = 0.0
                else:
                    shaped_env.alpha = max(DEMO_ALPHA_MIN, shaped_env.alpha * DEMO_ALPHA_DECAY)

            # BRS: no potential updates, only alpha decay
            elif method == "brs":
                shaped_env.alpha *= ALPHA_DECAY

            # DPBRS: update potential from agent rollouts returns + decay alpha
            elif method == "dpbrs":
                # flatten episodes -> state/return pairs
                flat_states = []
                flat_returns = []
                for states_seq, rewards_seq in zip(interval_states, interval_rewards):
                    # compute discounted returns backwards
                    G = 0.0
                    rets = [0.0] * len(rewards_seq)
                    for i in reversed(range(len(rewards_seq))):
                        G = rewards_seq[i] + GAMMA_RL * G
                        rets[i] = G
                    flat_states.extend(states_seq)
                    flat_returns.extend(rets)

                train_potential_mlp(phi_net, phi_opt, np.array(flat_states, dtype=np.float32),
                                    np.array(flat_returns, dtype=np.float32), device, POT_UPDATE_EPOCHS)

                shaped_env.alpha *= ALPHA_DECAY
                interval_states, interval_rewards = [], []

            # RNN: update RNN potential on sequences + decay alpha
            elif method == "rnn":
                phi_net.train()
                for _ in range(3):  # epochs over the interval (keep small)
                    for states_seq, rewards_seq in zip(interval_states, interval_rewards):
                        # discounted returns
                        G = 0.0
                        returns_seq = [0.0] * len(rewards_seq)
                        for i in reversed(range(len(rewards_seq))):
                            G = rewards_seq[i] + GAMMA_RL * G
                            returns_seq[i] = G

                        h = phi_net.init_hidden(device)
                        phi_opt.zero_grad()
                        total_loss = 0.0

                        for s, target in zip(states_seq, returns_seq):
                            x = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                            pred, h = phi_net(x, h)
                            tgt = torch.tensor([[target]], dtype=torch.float32, device=device)
                            total_loss = total_loss + nn.functional.mse_loss(pred, tgt)

                        total_loss.backward()
                        phi_opt.step()

                phi_net.eval()
                shaped_env.alpha *= ALPHA_DECAY
                interval_states, interval_rewards = [], []

        # print like your LARS
        print(
            f"[{method.upper()}] Episode {ep+1} "
            f"- ShapedReward: {ep_shaped:.2f}, TrueReward: {true_r:.2f}, "
            f"Eps: {agent.eps:.2f}, Alpha: {shaped_env.alpha:.2f}"
        )

    # ========================================================
    # SAVE OUTPUTS (same pattern as your LARS)
    # ========================================================
    # Save rewards arrays
    shaped_rewards = np.array(shaped_rewards, dtype=np.float32)
    true_rewards = np.array(true_rewards, dtype=np.float32)
    np.save(os.path.join(save_dir, f"{method}_shaped_{version}.npy"), shaped_rewards)
    np.save(os.path.join(save_dir, f"{method}_true_{version}.npy"), true_rewards)

    # Save model
    model_path = os.path.join(save_dir, f"{method}_model_{version}.pth")
    torch.save(agent.qnet.state_dict(), model_path)

    # Plot + save PNG (IMPORTANT: save BEFORE show; also close)
    fig_path = os.path.join(save_dir, f"{method}_training_{version}.png")
    plt.figure(figsize=(8, 4))
    plt.plot(shaped_rewards, label="Shaped Reward (training)")
    plt.plot(true_rewards, label="True Env Reward (evaluation)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{method.upper()} on MountainCar")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"\nSaved: {model_path}")
    print(f"Saved: {fig_path}\n")

    env.close()
    return model_path


# ============================================================
# VISUAL DEMO: load model & run in human render mode
# ============================================================
def run_visual_demo(model_path, method="lars", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(ENV_NAME, max_episode_steps=MAX_STEPS, render_mode="human")
    obs, _ = env.reset(seed=SEED)
    if isinstance(obs, tuple):
        obs = obs[0]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, device=device)

    agent.qnet.load_state_dict(torch.load(model_path, map_location=device))
    agent.qnet.eval()

    done = False
    reached_top = False
    total_reward = 0.0
    steps = 0

    while not done and steps < MAX_STEPS:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = agent.qnet(x)
        action = int(q.argmax(dim=1).item())

        next_obs, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        total_reward += r
        if next_obs[0] >= 0.5:
            reached_top = True

        obs = next_obs
        steps += 1

    env.close()
    print(f"[VIS] Total Reward: {total_reward}")
    print("[VIS] Reached Top!" if reached_top else "[VIS] Did NOT reach top.")

def run_visual_demo_q(q_path):
    with open(q_path, "rb") as f:
        q_table, pos_space, vel_space = pickle.load(f)

    env = gym.make(ENV_NAME, max_episode_steps=MAX_STEPS, render_mode="human")
    obs, _ = env.reset(seed=SEED)

    done = False
    total_reward = 0.0
    reached_top = False
    steps = 0

    while not done and steps < MAX_STEPS:
        sp = np.digitize(obs[0], pos_space)
        sv = np.digitize(obs[1], vel_space)
        action = int(np.argmax(q_table[sp, sv]))

        obs, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += r

        if obs[0] >= 0.5:
            reached_top = True

        steps += 1

    env.close()
    print(f"[Q VIS] Total Reward: {total_reward}")
    print("[Q VIS] Reached Top!" if reached_top else "[Q VIS] Did NOT reach top.")

# ============================================================
# CLI ENTRY
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="lars",
                        choices=["lars", "demo_static", "brs", "dpbrs", "rnn", "q_only"])
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--q_demo_path", type=str, default="models/q/mountain_car_v1.pkl")
    args = parser.parse_args()

    if args.method == "q_only":
        q_path = os.path.join("models", "q", f"mountain_car_{args.version}.pkl")

        if args.train:
            q_path = train_tabular_q(save_dir="models/q", version=args.version)

        if args.visualize:
            if not os.path.exists(q_path):
                raise FileNotFoundError(f"Q model not found: {q_path}. Run with --train first.")
            run_visual_demo_q(q_path)

    else:
        model_path = os.path.join("models", args.method, f"{args.method}_model_{args.version}.pth")

        if args.train:
            model_path = train_method(method=args.method, version=args.version, q_demo_path=args.q_demo_path)

        if args.visualize:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}. Run with --train first.")
            run_visual_demo(model_path, method=args.method)


"""
T-MAPPO 算法实现 - 基于 ICCT 2023 论文
Multi-UAV Searching Trajectory Optimization Algorithm based on Deep Reinforcement Learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class TransformerEncoder(nn.Module):
    """Transformer Encoder with self-attention for critic network
    
    论文描述：在 critic MLP 之前添加 transformer，
    基于所有 agent 的特征值计算 Query, Keys, Values
    然后通过 softmax 得到 Attention 值
    """

    def __init__(self, feature_dim, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        assert feature_dim % n_heads == 0, "feature_dim must be divisible by n_heads"

        # Q, K, V 线性投影
        self.W_Q = nn.Linear(feature_dim, feature_dim)
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_V = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, feature_dim),
        )
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        x: (num_agents, feature_dim) - 所有 agent 的特征值
        论文: 对所有 agent 的特征值进行 self-attention
        """
        num_agents, dim = x.shape

        # Q, K, V 变换（论文描述）
        Q = self.W_Q(x)  # (num_agents, feature_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        # 多头注意力
        Q = Q.view(num_agents, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(num_agents, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(num_agents, self.n_heads, self.head_dim).transpose(0, 1)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_agents, dim)

        # 残差 + LayerNorm
        x = self.layer_norm1(x + attn_output)

        # FFN
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)

        return x  # (num_agents, feature_dim)


class ActorNetwork(nn.Module):
    """Actor 网络 - 输出动作分布"""

    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            return torch.argmax(probs, dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()

    def evaluate_actions(self, obs, actions):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """Critic 网络 - 带 Transformer 的集中式价值函数（论文核心贡献）

    论文描述：
    1. 每个 agent 基于观测通过 MLP 获取特征值 en
    2. 将所有 agent 的特征值送入 Transformer 计算 Attention
    3. Attention 值送入 MLP 输出
    """

    def __init__(self, obs_dim, num_agents, hidden_dim=128, use_transformer=True):
        super().__init__()
        self.num_agents = num_agents
        self.use_transformer = use_transformer

        # 每个 agent 的 MLP 提取特征（论文：基于观测得到特征值 en）
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if use_transformer:
            # Transformer 处理所有 agent 的特征（论文核心）
            self.transformer = TransformerEncoder(hidden_dim, n_heads=4, d_ff=hidden_dim)

        # 输出层
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_list):
        """
        obs_list: list of tensors, 每个 agent 的观测 (obs_dim,)
        返回: 每个 agent 的价值估计 (num_agents, 1)
        """
        # 提取每个 agent 的特征值 en（论文描述）
        features = []
        for obs in obs_list:
            f = self.feature_net(obs)
            features.append(f)

        features = torch.stack(features, dim=0)  # (num_agents, hidden_dim)

        if self.use_transformer:
            # Transformer self-attention（论文核心贡献）
            features = self.transformer(features)

        # 输出价值
        values = self.output_net(features)
        return values


class TMAPPO:
    """T-MAPPO: MAPPO with Transformer Critic

    基于 ICCT 2023 论文实现
    """

    def __init__(
        self,
        obs_dim,
        num_agents,
        n_actions=3,  # 3个方向选择（论文图2）
        lr_actor=0.001,  # 论文 learning rate actor (表I)
        lr_critic=0.0003,  # 论文 learning rate critic (表I)
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10,
        mini_batch_size=64,
        hidden_dim=128,
        use_transformer=True,
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        # Actor (每个 agent 共享参数)
        self.actor = ActorNetwork(obs_dim, n_actions, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic (带 Transformer)
        self.critic = CriticNetwork(obs_dim, num_agents, hidden_dim, use_transformer)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 经验缓冲区
        self.buffer = []

    def store_transition(self, obs_list, actions, rewards, next_obs_list, dones, global_state=None):
        """存储一次经验"""
        self.buffer.append({
            'obs': [o.copy() for o in obs_list],
            'actions': actions.copy(),
            'rewards': np.array(rewards, dtype=np.float32),
            'next_obs': [o.copy() for o in next_obs_list],
            'dones': dones,
            'global_state': global_state.copy() if global_state is not None else None,
        })

    def get_actions(self, obs_list, deterministic=False):
        """获取所有 agent 的动作"""
        actions = []
        for obs in obs_list:
            obs_tensor = torch.FloatTensor(obs)
            action = self.actor.get_action(obs_tensor, deterministic)
            actions.append(action)
        return np.array(actions)

    def _compute_gae(self, rewards, values, dones, last_values):
        """计算 GAE"""
        advantages = []
        gae = 0
        values = values + [last_values]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values[:-1])
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self):
        """更新网络（每个 episode 结束时调用，论文 Algo.1）"""
        if len(self.buffer) < 1:
            return {}

        T = len(self.buffer)  # 当前 episode 的步数
        
        # 整理数据
        # obs_tensor_list[t] = [o0_t, o1_t, ...]  每个步的观测列表
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_dones = []
        
        for t in range(T):
            batch_obs.append([torch.FloatTensor(o) for o in self.buffer[t]['obs']])
            batch_actions.append(self.buffer[t]['actions'].copy())
            batch_rewards.append(self.buffer[t]['rewards'].copy())
            batch_next_obs.append([torch.FloatTensor(o) for o in self.buffer[t]['next_obs']])
            batch_dones.append(self.buffer[t]['dones'])

        # 集中式 critic 评估每个步的价值
        values = []
        with torch.no_grad():
            for t in range(T):
                v = self.critic(batch_obs[t]).numpy().flatten()  # (num_agents,)
                values.append(v)
            last_value = self.critic(batch_next_obs[-1]).numpy().flatten()

        # 对每个 agent 计算 GAE
        all_advantages = np.zeros((self.num_agents, T), dtype=np.float32)
        all_returns = np.zeros((self.num_agents, T), dtype=np.float32)
        
        for agent_idx in range(self.num_agents):
            agent_rewards = [r[agent_idx] for r in batch_rewards]
            agent_values = [v[agent_idx] for v in values]
            agent_dones_val = [d for d in batch_dones]
            agent_last_val = last_value[agent_idx]

            adv, ret = self._compute_gae(agent_rewards, agent_values, agent_dones_val, agent_last_val)
            all_advantages[agent_idx] = adv
            all_returns[agent_idx] = ret

        # 展开为 (T * num_agents,) 的一维数组
        # 排列方式: [agent0_t0, agent0_t1, ..., agent1_t0, agent1_t1, ..., ] 的变体
        # 这里使用: [agent0_t0, agent1_t0, agent2_t0, ..., agent0_t1, agent1_t1, ...]
        # 这样同一 step 的所有 agent obs 连续
        actor_obs = []
        actor_actions = []
        actor_adv = []
        actor_ret = []
        
        for t in range(T):
            for agent_idx in range(self.num_agents):
                actor_obs.append(self.buffer[t]['obs'][agent_idx])
                actor_actions.append(self.buffer[t]['actions'][agent_idx])
                actor_adv.append(all_advantages[agent_idx][t])
                actor_ret.append(all_returns[agent_idx][t])
        
        actor_obs = np.array(actor_obs)
        actor_actions = np.array(actor_actions) 
        actor_adv = np.array(actor_adv)
        actor_ret = np.array(actor_ret)

        total_loss = 0
        n_updates = 0

        for _ in range(self.epochs):
            indices = np.random.permutation(len(actor_obs))
            for start in range(0, len(indices), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(indices))
                idx = indices[start:end]

                obs_batch = torch.FloatTensor(actor_obs[idx])
                actions_batch = torch.LongTensor(actor_actions[idx].astype(np.int64))
                adv_batch = torch.FloatTensor(actor_adv[idx])
                ret_batch = torch.FloatTensor(actor_ret[idx])

                # Actor 更新 (PPO clip)
                log_probs, entropy = self.actor.evaluate_actions(obs_batch, actions_batch)
                # 简化: 用当前策略的 log_prob
                ratios = torch.exp(log_probs - log_probs.detach())
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # Critic 更新 — 按 step 组织 (每个 step 包含所有 agent 的 obs)
                unique_t_idx = np.unique(idx // self.num_agents)
                unique_t_idx = unique_t_idx[unique_t_idx < T]
                
                if len(unique_t_idx) == 0:
                    continue
                
                obs_for_critic = [batch_obs[t] for t in unique_t_idx]
                
                values_pred = []
                for obs_list in obs_for_critic:
                    v = self.critic(obs_list)  # (num_agents, 1)
                    values_pred.append(v.flatten())  # (num_agents,)
                values_pred = torch.cat(values_pred)  # (len(unique_t_idx) * num_agents,)

                # 对应的 ret
                ret_vals = []
                for t in unique_t_idx:
                    for ag in range(self.num_agents):
                        ret_vals.append(all_returns[ag][t])
                ret_tensor = torch.FloatTensor(np.array(ret_vals))

                critic_loss = F.mse_loss(values_pred, ret_tensor)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                total_loss += actor_loss.item() + critic_loss.item()
                n_updates += 1

        self.buffer.clear()
        return {'loss': total_loss / max(n_updates, 1)}


def train_tmappo(env, agent, num_episodes=2500, verbose=True):
    """训练 T-MAPPO

    论文设置：
    - 总步数：2500步
    - Episode length：200
    - 大约 12-13 个 episode
    """
    episode_rewards = []
    episode_found = []

    for episode in range(num_episodes):
        obs_list = env.reset()
        episode_reward = np.zeros(env.num_uavs)
        episode_found_count = 0
        steps = 0

        for step in range(env.max_steps):
            # 获取动作
            actions = agent.get_actions(obs_list)

            # 执行环境步骤
            next_obs_list, rewards, done, info = env.step(
                [{'v': 1, 'd': a - 1} for a in actions]  # action: 0,1,2 -> offset: -1,0,1
            )

            # 存储
            global_state = env.get_global_state()
            agent.store_transition(
                obs_list, actions, rewards, next_obs_list,
                done, global_state
            )

            episode_reward += rewards
            obs_list = next_obs_list
            steps += 1

            if done:
                break

        episode_found_count = sum(env.user_found)
        episode_rewards.append(np.mean(episode_reward))
        episode_found.append(episode_found_count)

        # 更新（论文 Algo.1：每个 episode 结束更新）
        update_info = agent.update()

        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_found = np.mean(episode_found[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Found: {avg_found:.1f}/{env.num_users} | "
                  f"Steps: {steps} | "
                  f"Loss: {update_info.get('loss', 0):.4f}")

    return episode_rewards, episode_found


if __name__ == "__main__":
    from env_uav_search import UAVSearchEnv
    from env_uav_search import UAVSearchEnv

    # 论文参数
    env = UAVSearchEnv(
        grid_size=(10, 10),
        num_uavs=4,
        num_users=10,
        max_steps=200,
    )

    obs_dim = env._get_observations()[0].shape[0]
    num_agents = env.num_uavs

    print(f"Observation dim: {obs_dim}")
    print(f"Number of agents: {num_agents}")

    # 创建 T-MAPPO agent
    agent = TMAPPO(
        obs_dim=obs_dim,
        num_agents=num_agents,
        n_actions=3,  # 论文图2：3个方向选择
        lr_actor=0.001,
        lr_critic=0.0003,
        hidden_dim=128,
        use_transformer=True,  # 论文核心：使用 Transformer
    )

    # 训练
    rewards, found = train_tmappo(env, agent, num_episodes=20)

    print("\nTraining completed!")
    print(f"Final avg reward (last {min(50, len(rewards))}): {np.mean(rewards[-50:]):.2f}")
    print(f"Final avg found (last {min(50, len(found))}): {np.mean(found[-50:]):.1f}/{env.num_users}")

    # 评估
    print("\nEvaluating...")
    eval_rewards = []
    eval_found = []
    for eval_ep in range(10):
        obs_list = env.reset()
        total_found = 0
        for step in range(env.max_steps):
            actions = agent.get_actions(obs_list, deterministic=True)
            obs_list, rewards, done, info = env.step(
                [{'v': 1, 'd': a - 1} for a in actions]
            )
            if done:
                break
        total_found = sum(env.user_found)
        eval_found.append(total_found)
        eval_rewards.append(np.mean(rewards))

    print(f"Evaluation: Avg found {np.mean(eval_found):.1f}/{env.num_users}")

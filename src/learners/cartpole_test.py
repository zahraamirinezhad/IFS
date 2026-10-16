import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Args:
    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2
    use_adv_std = True
    start_entropy_coef = 0.02
    min_entropy_coef = 0.001
    entropy_anneal_time = 10000
    grad_norm_clip = 0.5
    critic_grad_norm_clip = 0.5
    use_huber_loss = True
    huber_delta = 1.0
    clip_value_loss = True
    value_loss_coef = 0.5
    use_popart = False
    epochs = 4

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x)

class SingleAgentPPOTrainer:
    def __init__(self, state_dim, action_dim):
        self.args = Args()
        self.actor = PolicyNet(state_dim, action_dim)
        self.critic = ValueNet(state_dim)
        
        self.agent_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())
        
        self.agent_optimiser = th.optim.Adam(self.agent_params, lr=1e-3)
        self.critic_optimiser = th.optim.Adam(self.critic_params, lr=1e-3)

    def _zero_shared_grads(self):
        pass

    def gae_target(self, rewards, mask, values, terminated):
        B, T_rew, N = rewards.shape[0], rewards.shape[1], values.shape[2]
        gae = th.zeros((B, N), device=rewards.device)
        advantages = th.zeros((B, T_rew, N), device=rewards.device)

        for step in reversed(range(T_rew)):
            non_terminal = 1.0 - terminated[:, step].float()
            valid_step = mask[:, step].float()
            next_values = values[:, step + 1]
            delta = rewards[:, step] + self.args.gamma * next_values * non_terminal - values[:, step]
            gae = (delta + self.args.gamma * self.args.gae_lambda * non_terminal * gae) * valid_step
            advantages[:, step] = gae

        targets = advantages + values[:, :T_rew]
        return advantages.detach(), targets.detach()

    def actor_update(self, log_pi_taken, entropy, advantages, old_log_prob_taken, mask, actor_train_stats, t_env=0):
        valid_mask = mask == 1
        if not valid_mask.any(): return actor_train_stats

        if getattr(self.args, "use_adv_std", True):
            valid_advs = advantages[valid_mask]
            if valid_advs.numel() > 1:
                mean = valid_advs.mean()
                std = valid_advs.std()
                advantages = (advantages - mean) / (std + 1e-8)

        flat_log_pi = log_pi_taken[valid_mask]
        flat_old_log_pi = old_log_prob_taken[valid_mask].detach()
        flat_adv = advantages[valid_mask].detach()
        flat_entropy = entropy[valid_mask]

        log_ratios = flat_log_pi - flat_old_log_pi
        ratios = th.exp(log_ratios)

        surr1 = ratios * flat_adv
        surr2 = th.clamp(ratios, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip) * flat_adv
        policy_loss = -th.min(surr1, surr2).mean()
        entropy_loss = flat_entropy.mean()

        start_entropy_coef = self.args.start_entropy_coef
        min_entropy_coef = self.args.min_entropy_coef
        anneal_ratio = min(1.0, float(t_env) / float(self.args.entropy_anneal_time))
        current_entropy_coef = start_entropy_coef - anneal_ratio * (start_entropy_coef - min_entropy_coef)

        actor_loss = policy_loss - current_entropy_coef * entropy_loss

        self.agent_optimiser.zero_grad()
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()
        return actor_train_stats

    def critic_update(self, curr_values, old_values, target_returns, mask, critic_train_stats):
        valid_mask = mask == 1
        if not valid_mask.any(): return critic_train_stats

        curr_v = curr_values[valid_mask]
        old_v = old_values[valid_mask]
        targets = target_returns[valid_mask]

        td_error = targets - curr_v

        if self.args.use_huber_loss:
            loss_raw = F.huber_loss(curr_v, targets, delta=self.args.huber_delta, reduction="none")
        else:
            loss_raw = 0.5 * (td_error ** 2)

        if self.args.clip_value_loss:
            clipped_v = old_v + (curr_v - old_v).clamp(-self.args.eps_clip, self.args.eps_clip)
            if self.args.use_huber_loss:
                clipped_loss_raw = F.huber_loss(clipped_v, targets, delta=self.args.huber_delta, reduction="none")
            else:
                clipped_loss_raw = 0.5 * ((targets - clipped_v) ** 2)
            loss = th.max(loss_raw, clipped_loss_raw).mean()
        else:
            loss = loss_raw.mean()

        total_critic_loss = loss * self.args.value_loss_coef

        self.critic_optimiser.zero_grad()
        total_critic_loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic_params, self.args.critic_grad_norm_clip)
        self.critic_optimiser.step()
        return critic_train_stats

    def train_on_batch(self, states, actions, rewards, next_states, dones, t_env):
        # اصلاح تبدیل لیست‌ها به NumPy Array قبل از تبدیل به Tensor برای افزایش سرعت
        s = th.tensor(np.array(states), dtype=th.float32).unsqueeze(0)
        a = th.tensor(np.array(actions), dtype=th.long).unsqueeze(0).unsqueeze(-1)
        r = th.tensor(np.array(rewards), dtype=th.float32).unsqueeze(0).unsqueeze(-1)
        s_next = th.tensor(np.array(next_states), dtype=th.float32).unsqueeze(0)
        d = th.tensor(np.array(dones), dtype=th.float32).unsqueeze(0).unsqueeze(-1)
        
        T = len(states)
        mask = th.ones((1, T, 1))

        with th.no_grad():
            probs = self.actor(s)
            dist = th.distributions.Categorical(probs)
            old_log_probs = dist.log_prob(a.squeeze(-1)).unsqueeze(-1)
            
            all_s = th.cat([s, s_next[:, -1:]], dim=1)
            values = self.critic(all_s) # [1, T+1, 1]

        advantages, target_returns = self.gae_target(r, mask, values, d)

        # جداسازی values از گراف قبل از ورود به اپوک‌ها
        old_values = values[:, :-1].detach()

        # Epochs آموزش
        for _ in range(self.args.epochs):
            probs_curr = self.actor(s)
            dist_curr = th.distributions.Categorical(probs_curr)
            log_pi = dist_curr.log_prob(a.squeeze(-1)).unsqueeze(-1)
            entropy = dist_curr.entropy().unsqueeze(-1)
            
            curr_v = self.critic(s)

            self.actor_update(log_pi, entropy, advantages, old_log_probs, mask, {}, t_env)
            self.critic_update(curr_v, old_values, target_returns, mask, {})

def run_cartpole_test():
    env = gym.make("CartPole-v1")
    trainer = SingleAgentPPOTrainer(env.observation_space.shape[0], env.action_space.n)
    
    total_steps = 0
    print("\n--- شروع تست همگرایی روی CartPole-v1 ---")
    
    for episode in range(1, 201):
        state, _ = env.reset()
        done = False
        states, actions, rewards, next_states, dones = [], [], [], [], []
        ep_reward = 0

        while not done:
            s_tensor = th.tensor(np.array(state), dtype=th.float32).unsqueeze(0)
            with th.no_grad():
                probs = trainer.actor(s_tensor)
                action = th.distributions.Categorical(probs).sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            ep_reward += reward
            total_steps += 1

        trainer.train_on_batch(states, actions, rewards, next_states, dones, total_steps)

        if episode % 10 == 0:
            print(f"Episode {episode:3d} | Total Steps: {total_steps:5d} | Reward: {ep_reward:.1f}")

        if ep_reward >= 450:
            print(f"\n✅ [Test Successful] الگوریتم در اپیزود {episode} به پاداش عالی ({ep_reward}) همگرا شد!")
            break

if __name__ == "__main__":
    run_cartpole_test()
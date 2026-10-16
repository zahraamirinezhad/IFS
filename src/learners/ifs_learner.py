# code heavily adapted from original ePymarl implementation, based on the MAPPO implementation.
import os
import numpy as np
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from controllers import AgentOwnedMAC, OpenTrainMAC
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry

from components.ifs.ifs import extract_visibility_mask, build_pairwise_encoder_inputs

from utils.mappo_util import huber_loss
from utils.rl_utils import compute_per_agent_terminated_mask, get_noop_act


class IFSLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, 
                                   lr=args.lr,
                                   eps=self.args.optim_eps
                                   )
        self.critic = critic_registry[args.critic_type](scheme, args)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params,
                                   lr=getattr(args, "critic_lr", args.critic_lr),
                                   eps=self.args.optim_eps
                                   )
        
        self.ifs_params = list(mac.parameters())
        self.ifs_optimiser = Adam(params=self.ifs_params, lr=getattr(args, "ifs_lr", args.lr), eps=self.args.optim_eps)

        if isinstance(self.mac, AgentOwnedMAC) or isinstance(self.mac, OpenTrainMAC):
            pass
        else:
            raise Exception("ERROR: mac type not supported for IFS")

        self.log_stats_t = 0
        self.log_extra_stats_t = 0
        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
        
        self.value_normalizer = self.critic.value_normalizer
        self.noop_acts = get_noop_act(args, device)

        self.extra_stats_path = os.path.join(args.local_results_path, "data", args.expt_logname)
        os.makedirs(self.extra_stats_path, exist_ok=True)


    def compute_mask(self, batch, max_t):
        if self.args.mask_type == "team":
            terminated = batch["terminated"][:, :max_t].float()
            mask = batch["filled"][:, :max_t].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            mask = mask.repeat(1, 1, self.n_agents) 
        elif self.args.mask_type == "agent":
            mask = compute_per_agent_terminated_mask(terminated_batch=batch['terminated'][:, :max_t],
                                                     avail_actions_batch= batch['avail_actions'][:, :max_t], 
                                                     n_rollout_threads=batch.batch_size,
                                                     no_op_tensor_list=self.noop_acts,
                                                     ep_limit=max_t, device=self.args.device
                                                     )
        else: 
            raise Exception("ERROR: mask_type {} not implemented".format(self.args.mask_type))
        # mask shape: (bs, ep_len, n_agents)
        return mask

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        max_t = batch['terminated'].shape[1] - 1
        actor_mask = self.compute_mask(batch, max_t=max_t)
        critic_mask = actor_mask.detach().clone()
        ed_mask = actor_mask.detach().clone()
        
        if self.args.open_train_or_eval:
            if self.args.ed_model_uncontrolled_only:
                ed_mask = ed_mask * (~batch["trainable_agents"][:, :max_t].squeeze(-1))
            else: 
                ed_mask = ed_mask * batch["trainable_agents"][:, :max_t].squeeze(-1)
            if self.args.trainable_agents_mask_actor:
                actor_mask = actor_mask * batch["trainable_agents"][:, :max_t].squeeze(-1)
            if self.args.trainable_agents_mask_critic:
                critic_mask = critic_mask * batch["trainable_agents"][:, :max_t].squeeze(-1)

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
            
        old_values, old_critic_hidden_states = self.critic_forward_nograd(self.critic, batch)
        old_values = old_values[:, :-1].squeeze(3)
        old_critic_hidden_states = old_critic_hidden_states

        target_returns = self.compute_target_returns(self.critic, 
                                                   batch, old_critic_hidden_states, 
                                                   rewards).clone().detach()
        
        if self.value_normalizer:
            self.value_normalizer.update(target_returns)
            denorm_old_values = self.value_normalizer.denormalize(old_values)
        else:
            denorm_old_values = old_values
        advantages = (target_returns - denorm_old_values)

        if self.args.use_adv_std:
            advantages_copy = advantages.clone().detach()
            std_advantages, mean_advantages = th.std_mean(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        old_log_prob_taken, _ = self.actor_forward_all(self.mac, batch, actions)
        old_log_prob_taken = old_log_prob_taken.clone().detach()

        actor_train_stats = {"actor_loss": [], "actor_grad_norm": [], "entropy": [], "log_pi_taken": [], "ratio": []}
        critic_train_stats = {"critic_loss": [], "critic_grad_norm": [], "td_error_abs": [], "target_mean": [], "curr_taken_v": []}
        ifs_train_stats = {"ifs_recon_loss": [], "ifs_grad_norm": []}

        # مرحله پیش‌آموزش ماژول‌های IFS (مشابه POAM)
        pretrain_ts = getattr(self.args, "ed_pretrain_ts", 0)
        pretrain_epochs = getattr(self.args, "ed_pretrain_epochs", 10)

        if t_env <= pretrain_ts:
            print(f"Training IFS auxiliary modules only. t_env is {t_env}")
            for _ in range(pretrain_epochs):
                ifs_train_stats = self.ifs_auxiliary_update(batch=batch,
                                                            mask=ed_mask,
                                                            ifs_train_stats=ifs_train_stats,
                                                            t_env=t_env)
        else: 
            print(f"Switching from updating IFS auxiliary only to updating all. t_env is {t_env}")

            for _ in range(self.args.epochs):
                mb_rand = th.randperm(batch.batch_size).numpy() 
                mb_size = batch.batch_size // self.args.n_minibatch
                sampler = [np.array(mb_rand[i * mb_size:(i + 1) * mb_size]) for i in range(self.args.n_minibatch)]
                old_ed_hidden, old_critic_hidden = old_critic_hidden_states

                for indices in sampler:             
                    log_pi_taken, entropy = self.actor_forward_all(self.mac, batch[indices], actions[indices])
                    curr_v = self.critic_forward_all(self.critic, 
                                                   batch[indices], 
                                                   hidden_states=(old_ed_hidden[indices], old_critic_hidden[indices]))
                    curr_v = curr_v[:, :-1].squeeze(3)
                    
                    actor_train_stats = self.actor_update(log_pi_taken=log_pi_taken, 
                                                            entropy=entropy, 
                                                            advantages=advantages[indices],
                                                            old_log_prob_taken=old_log_prob_taken[indices], 
                                                            mask=actor_mask[indices],
                                                            actor_train_stats=actor_train_stats)
                    
                    critic_train_stats = self.critic_update(curr_values=curr_v,
                                                              old_values=old_values[indices], 
                                                              target_returns=target_returns[indices],
                                                              mask=critic_mask[indices], 
                                                              critic_train_stats=critic_train_stats)
                    
                    ifs_train_stats = self.ifs_auxiliary_update(batch=batch[indices],
                                                                mask=ed_mask[indices],
                                                                ifs_train_stats=ifs_train_stats,
                                                                t_env=t_env)

        # logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval or self.log_stats_t == 0:
            ts_logged = len(critic_train_stats["critic_loss"]) if len(critic_train_stats["critic_loss"]) > 0 else 1
            for key in critic_train_stats:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)
            ts_logged = len(actor_train_stats["actor_loss"]) if len(actor_train_stats["actor_loss"]) > 0 else 1
            for key in actor_train_stats:
                self.logger.log_stat(key, sum(actor_train_stats[key]) / ts_logged, t_env)
            ts_logged = len(ifs_train_stats["ifs_recon_loss"]) if len(ifs_train_stats["ifs_recon_loss"]) > 0 else 1
            for key in ifs_train_stats:
                self.logger.log_stat(key, sum(ifs_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("old_values_mean", old_values.mean().item(), t_env)
            self.logger.log_stat("advantage_mean", advantages.mean().item(), t_env)
            self.log_stats_t = t_env

    def reshape_batches(self, batch_list, bs, max_t):
        '''Reshape data from (bs, ep_len, n_agents, feat_size) to (bs * ep_len, n_agents, feat_size)'''
        reshaped = []
        for batch in batch_list:
            reshaped.append(batch.reshape(bs * max_t, -1))
        return tuple(reshaped)
    
    def ifs_auxiliary_update(self, batch, mask, ifs_train_stats, t_env=None):
        obs = batch["obs"] # [bs, max_t + 1, n_agents, obs_dim]
        vis_mask = extract_visibility_mask(obs, self.n_agents, availability_prob=getattr(self.args, "availability_prob", 0.7))
        
        enc_inputs = build_pairwise_encoder_inputs(self.n_agents, batch, t=None) # [bs, max_t, n_agents, n_others, feat_dim]
        bs, max_t, n_agents, n_others, feat_sz = enc_inputs.shape
        
        enc_inputs_flat = enc_inputs[:, :-1].reshape(-1, n_others, feat_sz)
        
        if vis_mask is not None:
            masked_list = []
            for i in range(self.n_agents):
                others_idx = [j for j in range(self.n_agents) if j != i]
                agent_mask = vis_mask[:, :-1, i, others_idx]
                masked_list.append(agent_mask.unsqueeze(2))
            vis_mask_paired = th.cat(masked_list, dim=2)
            flat_mask = vis_mask_paired.reshape(-1, n_others)
        else:
            flat_mask = None

        ifs_arch = self.mac.trained_agent.policy.ifs_architecture
        e_i = ifs_arch.ifm(enc_inputs_flat, flat_mask)
        reconstructed_slots = ifs_arch.ism(e_i, flat_mask) # [bs*max_t*n_agents, n_train_slots, agent_feature_dim]

        target_records = enc_inputs[:, :-1].reshape(-1, n_others, feat_sz)
        if flat_mask is not None:
            if flat_mask.shape[1] != target_records.shape[1]:
                flat_mask = flat_mask[:, :target_records.shape[1]]
            target_records = target_records * flat_mask.unsqueeze(-1).float()

        reconstructed_slots = reconstructed_slots[:, :target_records.shape[1], :target_records.shape[-1]]
        recon_loss = F.mse_loss(reconstructed_slots, target_records, reduction='sum')
        
        valid_elems = flat_mask.sum() if flat_mask is not None else th.tensor(target_records.numel(), device=target_records.device)
        loss = recon_loss / (valid_elems + 1e-8)

        self.ifs_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.ifs_params, getattr(self.args, "grad_norm_clip", 10.0))
        self.ifs_optimiser.step()

        ifs_train_stats['ifs_recon_loss'].append(loss.item())
        ifs_train_stats['ifs_grad_norm'].append(grad_norm.item())
        return ifs_train_stats
    
    def actor_update(self, log_pi_taken, entropy, advantages, old_log_prob_taken, mask, actor_train_stats):
        mb, max_t = log_pi_taken.shape[:2]
        rand = th.randperm(mb * max_t).numpy() 
        log_pi_taken = log_pi_taken.reshape(mb * max_t, -1)[rand]
        entropy = entropy.reshape(mb * max_t, -1)[rand]
        advantages = advantages.reshape(mb * max_t, -1)[rand]
        old_log_prob_taken = old_log_prob_taken.reshape(mb * max_t, -1)[rand]
        mask = mask.reshape(mb * max_t, -1)[rand]

        ratios = th.exp(log_pi_taken - old_log_prob_taken)
        surr1 = ratios * advantages  
        surr2 = th.clamp(ratios, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip) * advantages
        policy_loss = -th.min(surr1, surr2).mean()        
        actor_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

        self.agent_optimiser.zero_grad()
        actor_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        actor_train_stats["actor_loss"].append(actor_loss.item())
        actor_train_stats.setdefault("policy_loss", []).append(policy_loss.item())
        actor_train_stats["actor_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        actor_train_stats["entropy"].append(entropy.mean().item())
        actor_train_stats["log_pi_taken"].append(log_pi_taken.mean().item())
        actor_train_stats["ratio"].append((ratios * mask).sum().item() / mask_elems)
        return actor_train_stats

    @th.no_grad()
    def critic_forward_nograd(self, critic, batch): 
        v = []
        h_e, h = critic.init_hidden()
        critic_hidden_states = []
        ed_hidden_states = []
        for t in range(batch.max_seq_length):
            ed_hidden_states.append(h_e)
            critic_hidden_states.append(h)
            v_t, h_e, h = critic.forward(batch, (h_e, h.detach()), t=t)
            v.append(v_t)
        v = th.cat(v, dim=1) 
        ed_hidden_states = th.cat(ed_hidden_states, dim=1)
        critic_hidden_states = th.cat(critic_hidden_states, dim=1)
        return v, (ed_hidden_states, critic_hidden_states)
    
    def critic_forward_all(self, critic, batch, hidden_states):
        v, _, _ = critic.forward(batch, hidden_states, t=None)
        return v

    def actor_forward_all(self, mac, batch, actions):
        agent_outs, _ = mac.forward(batch, t=None)
        log_probs_all, entropy_all = mac.action_selector.eval_action(agent_inputs=agent_outs[:, :-1], 
                                                                   actions=actions.squeeze(-1))
        return log_probs_all, entropy_all
    
    def compute_target_returns(self, critic, batch, old_critic_hidden_states, rewards):
        '''
        old_critic_hidden_states: the hidden states computed by the critic 
        before any updates (i.e. at time data was gathered)
        '''
        # TODO: remove extra critic call 
        with th.no_grad(): 
            target_vals = self.critic_forward_all(critic, batch, hidden_states=old_critic_hidden_states)
            target_vals = target_vals.squeeze(3)

        target_mask = self.compute_mask(batch, max_t=batch['terminated'].shape[1])
        if self.value_normalizer is not None:
            target_vals = self.value_normalizer.denormalize(target_vals)

        if self.args.use_gae:
            target_returns = self.gae_target(rewards, target_mask, target_vals)
        else:
            target_returns = self.nstep_returns(rewards, target_mask, target_vals, self.args.q_nstep)
        
        return target_returns # shape (batch_size, episode_length, n_agents)

    def critic_update(self, curr_values, old_values, target_returns, mask, critic_train_stats):
        if self.value_normalizer is not None:
            target_returns = self.value_normalizer.normalize(target_returns)            

        mb, max_t = curr_values.shape[:2]
        rand = th.randperm(mb * max_t).numpy() 
        curr_values = curr_values.reshape(mb * max_t, -1)[rand]
        old_values = old_values.reshape(mb * max_t, -1)[rand]
        target_returns = target_returns.reshape(mb * max_t, -1)[rand]
        mask = mask.reshape(mb * max_t, -1)[rand]
        clipped_v = old_values + (curr_values - old_values).clamp(-self.args.eps_clip, self.args.eps_clip)

        td_error = (target_returns - curr_values)
        clipped_td_error = (target_returns - clipped_v)

        if self.args.use_huber_loss:
            loss = huber_loss(td_error, self.args.huber_delta)
            clipped_loss = huber_loss(clipped_td_error, self.args.huber_delta)
        else:
            loss = (td_error ** 2) / 2.0
            clipped_loss = (clipped_td_error ** 2) / 2.0

        if self.args.clip_value_loss:
            loss = th.max(loss, clipped_loss)
        
        loss = (loss * mask).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        (loss * 0.5).backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        critic_train_stats["critic_loss"].append(loss.item())
        critic_train_stats["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        critic_train_stats["td_error_abs"].append(((td_error * mask).abs().sum().item() / mask_elems))
        critic_train_stats["curr_taken_v"].append((curr_values * mask).sum().item() / mask_elems)
        critic_train_stats["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return critic_train_stats

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def gae_target(self, rewards, mask, values):
        # Returns is computed as the GAE +V_pi(s_t), which is equivalent to the TD lambda return
        # V_pi(s_t) = Q_pi(s_t, pi(s_t)) = A^GAE_pi(s_t, pi(s_t)) + V_pi(s_t)
        returns = []
        T = rewards.shape[1]
        gae = 0
        for step in reversed(range(T)):
            delta = rewards[:, step] + self.args.gamma * values[:, step + 1] * mask[:, step + 1] - \
                    values[:, step]
            gae = delta + self.args.gamma * self.args.gae_lambda * mask[:, step + 1] * gae
            returns.append((gae + values[:, step]).unsqueeze(1)) # list of len 52
        returns = th.flip(th.cat(returns, axis=1), dims=[1])
        return returns[:, :T]

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.ifs_optimiser.state_dict(), "{}/ifs_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.ifs_optimiser.load_state_dict(
            th.load("{}/ifs_opt.th".format(path), map_location=lambda storage, loc: storage))
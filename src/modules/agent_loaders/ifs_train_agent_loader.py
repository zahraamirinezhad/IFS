import torch as th
import torch.nn.functional as F

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from modules.agent_loaders.base_agent_loader import BaseAgentLoader

class IFSTrainAgentLoader(BaseAgentLoader):
    '''
    شبیه‌سازی شده مطابق POAMTrainAgentLoader برای مدیریت لود و اجرای عامل IFS
    '''
    def __init__(self, args, scheme, model_path):
        super().__init__(args, scheme,
                         n_agents=args.n_agents,
                         obs_last_action=args.obs_last_action,
                         obs_agent_id=args.obs_agent_id,
                         )
        
        input_shape = self._get_input_shape(scheme, args)
        self._build_agent(input_shape=input_shape)
        if model_path != "":
            raise NotImplementedError

        self.agent_output_type = self.args.agent_output_type
        self.action_selector = action_REGISTRY[self.args.action_selector](self.args)
    
    def predict(self, ep_batch, agent_idx_list, t_ep, t_env, bs, test_mode):
        ts = slice(None) if t_ep is None else slice(t_ep, t_ep+1)
        ret_t = ep_batch.max_seq_length if t_ep is None else 1
        ep_batch_sliced = ep_batch[bs]
        avail_actions = ep_batch["avail_actions"][bs, ts][:, :, agent_idx_list]
    
        if t_ep is None:
            h_e, h = self.policy.init_hidden(ep_batch.batch_size)
            h = h.to(self.args.device)
            h_e = h_e.to(self.args.device)
            agent_outs, hiddens, ed_hiddens = [], [], []
            for t in range(ep_batch_sliced.max_seq_length):
                out, h_e, h = self.policy(ep_batch, t, hidden_state=(h_e, h))
                agent_outs.append(out)
                ed_hiddens.append(h_e)
                hiddens.append(h)
            agent_outs = th.cat(agent_outs, dim=1)[bs]
            self.hidden_state = (th.cat(ed_hiddens, dim=1)[bs], th.cat(hiddens, dim=1)[bs])
        else: 
            if t_ep == 0 or not hasattr(self, "hidden_state"):
                h_e, h = self.policy.init_hidden(ep_batch.batch_size)
                h_e = h_e.to(self.args.device)
                h = h.to(self.args.device)
                self.hidden_state = (h_e, h)
            
            agent_outs, h_e, h = self.policy(ep_batch, t_ep, hidden_state=self.hidden_state)
            self.hidden_state = (h_e, h)
            agent_outs = agent_outs[bs, :][:, :, agent_idx_list]

        hidden_state = th.zeros(ep_batch_sliced.batch_size, ret_t, len(agent_idx_list), self.args.hidden_dim, device=self.args.device)
        
        ret_shape = (ep_batch_sliced.batch_size, ret_t, len(agent_idx_list), -1)
        agent_outs = agent_outs.reshape(*ret_shape)
        
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):            
                agent_outs[avail_actions == 0] = -1e10

        chosen_actions = self.action_selector.select_action(agent_outs, 
                                                           avail_actions,
                                                           t_env, 
                                                           test_mode=test_mode)
        chosen_actions = chosen_actions.reshape(*ret_shape)
        hidden_state = hidden_state.reshape(ep_batch_sliced.batch_size, ret_t, len(agent_idx_list), self.args.hidden_dim)
        return agent_outs, chosen_actions, hidden_state

    def _build_agent(self, input_shape):
        self.policy = agent_REGISTRY[self.args.agent](input_shape, self.args)    

    def set_modules(self, ifs_architecture):
        self.policy.set_modules(ifs_architecture)
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.mlp import MLPBase
from utils.mappo_util import init_rnn, init_module

from components.ifs.ifs import extract_visibility_mask, build_pairwise_encoder_inputs, IFSArchitectureWithConstraints

class IFSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(IFSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_size = input_shape
        
        if hasattr(self.args, "ed_hidden_dim"):
            self.hidden_dim = self.args.hidden_dim
            self.ed_hidden_dim = self.args.ed_hidden_dim
        else: 
            self.hidden_dim = self.ed_hidden_dim = self.args.hidden_dim  

        self.base = MLPBase(input_shape + args.embed_dim, 
                            self.hidden_dim, 
                            n_hidden_layers=1,
                            use_feature_norm=args.use_obs_norm, 
                            use_orthogonal=args.use_orthogonal_init
                            )
                            
        if self.args.use_rnn:
            self.rnn = init_rnn(nn.GRUCell(self.hidden_dim, self.hidden_dim), args.use_orthogonal_init)
            self.rnn_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.rnn = init_module(nn.Linear(self.hidden_dim, self.hidden_dim))
            
        self.fc2 = init_module(nn.Linear(self.hidden_dim, args.n_actions))
        
        obs_dim = input_shape - (args.n_agents if args.obs_agent_id else 0) - (args.n_actions if args.obs_last_action else 0)
        message_dim = getattr(args, "message_dim", 64)
        fused_dim = getattr(args, "fused_dim", 128)
        n_train_slots = getattr(args, "n_train_slots", args.n_agents - 1)
        
        pairwise_feat_dim = obs_dim + args.n_actions + args.n_agents * 2
        
        self.ifs_architecture = IFSArchitectureWithConstraints(
            args=args, 
            obs_dim=pairwise_feat_dim - message_dim, 
            message_dim=message_dim, 
            hidden_dim=self.ed_hidden_dim, 
            fused_dim=fused_dim, 
            n_train_slots=n_train_slots
        ) 

    def init_hidden(self, batch_size):
        encoder_hidden = self.base.mlp.fc1[0].weight.new(batch_size, 1, self.n_agents, self.ed_hidden_dim).zero_()
        hidden_base = self.base.mlp.fc1[0].weight.new(batch_size, 1, self.n_agents, self.hidden_dim).zero_()
        return encoder_hidden, hidden_base 

    def get_out_shapes(self, curr_t, batch_size, max_t):
        if curr_t is None:
            ts = slice(None)
            out_batch_dim = (batch_size, max_t, self.n_agents)
            hidden_batch_dim = (batch_size, max_t, self.n_agents, self.hidden_dim)
            ed_hidden_batch_dim = (batch_size, max_t, self.n_agents, self.ed_hidden_dim)
        else: 
            ts = slice(curr_t, curr_t + 1)
            out_batch_dim = (batch_size, 1, self.n_agents)
            hidden_batch_dim = (batch_size, 1, self.n_agents, self.hidden_dim)
            ed_hidden_batch_dim = (batch_size, 1, self.n_agents, self.ed_hidden_dim)
        return ts, out_batch_dim, hidden_batch_dim, ed_hidden_batch_dim

    def forward(self, ep_batch, t, hidden_state=None):
        ts, out_batch_dim, hidden_batch_dim, ed_hidden_batch_dim = self.get_out_shapes(
            curr_t=t, batch_size=ep_batch.batch_size, max_t=ep_batch.max_seq_length
        )
        inputs = self._build_inputs(ep_batch, t)
        inputs = inputs.reshape(-1, self.input_size)

        if hidden_state is None:
            hidden_state = ep_batch["actor_hidden_states"][:, ts]
            h_e, h_in = hidden_state[:, :, :, :, 0], hidden_state[:, :, :, :, 1]
        else: 
            h_e, h_in = hidden_state
            
        h_in = h_in.reshape(-1, self.hidden_dim)
        h_e = h_e.reshape(-1, self.ed_hidden_dim)
        
        vis_mask = extract_visibility_mask(ep_batch["obs"][:, ts], self.n_agents, availability_prob=getattr(self.args, "availability_prob", 0.7))

        enc_inputs = build_pairwise_encoder_inputs(self.n_agents, ep_batch, t)
        bs_flat, max_t_flat, n_ag, n_oth, feat_sz = enc_inputs.shape
        enc_inputs_flat = enc_inputs.reshape(-1, n_oth, feat_sz)
        
        if vis_mask is not None:
            masked_list = []
            for i in range(self.n_agents):
                others_idx = [j for j in range(self.n_agents) if j != i]
                agent_mask = vis_mask[:, :, i, others_idx]
                masked_list.append(agent_mask.unsqueeze(2))
            vis_mask_paired = th.cat(masked_list, dim=2)
            flat_mask = vis_mask_paired.reshape(-1, n_oth)
        else:
            flat_mask = None
        
        # flat_mask = vis_mask.reshape(-1, n_oth) if vis_mask is not None else None

        e_i, b_i, d_i, reconstructed_data = self.ifs_architecture(enc_inputs_flat, h_e, flat_mask)
        
        inputs = th.cat([e_i.detach(), inputs], dim=1)    
        x = F.relu(self.base(inputs))

        if self.args.use_rnn:
            h_out = self.rnn(x, h_in)
            h_norm = self.rnn_norm(h_out)
        else:
            h_norm = h_out = F.relu(self.rnn(x))
        
        q = self.fc2(h_norm)
        h_e = b_i.view(*ed_hidden_batch_dim)
        h_out = h_out.view(*hidden_batch_dim)
        return q.view(*out_batch_dim, -1), h_e, h_out

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        inputs.append(batch["obs"][:, ts]) 
        if self.args.obs_last_action:
            if t is None:
                last_act_all = th.cat([th.zeros_like(batch["actions_onehot"][:, [0]]), 
                                          batch["actions_onehot"][:, slice(0, max_t-1)]],
                                         dim=1)
                inputs.append(last_act_all) 
            else:
                last_act = th.zeros_like(batch["actions_onehot"][:, ts]) if t == 0 else batch["actions_onehot"][:, slice(t-1, t)]
                inputs.append(last_act)

        if self.args.obs_agent_id:
            agent_id_onehot = th.eye(self.n_agents, device=batch.device)
            agent_id_onehot = agent_id_onehot.expand(bs, max_t, self.n_agents, -1) if t is None \
                else agent_id_onehot.expand(bs, 1, self.n_agents, -1)
            inputs.append(agent_id_onehot)
            
        return th.cat(inputs, dim=-1)

    def set_modules(self, ifs_architecture):
        self.ifs_architecture = ifs_architecture

    def load(self, path):
        state_dict_all = th.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict_all)
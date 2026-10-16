import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.popart import PopArt
from utils.mappo_util import init_module, init_rnn
from utils.mlp import MLPBase

from components.ifs.ifs import build_pairwise_encoder_inputs, IFSArchitectureWithConstraints

class IFSCritic(nn.Module):
    def __init__(self, scheme, args):
        super(IFSCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_size = self._get_input_shape(scheme)
        
        if hasattr(self.args, "ed_hidden_dim"):
            self.hidden_dim = self.args.hidden_dim
            self.ed_hidden_dim = self.args.ed_hidden_dim
        else: 
            self.hidden_dim = self.ed_hidden_dim = self.args.hidden_dim  

        self.output_type = "v"
        
        self.base = MLPBase(self.input_size + args.embed_dim, 
                            self.hidden_dim, 
                            n_hidden_layers=1,
                            use_feature_norm=args.use_obs_norm, 
                            use_orthogonal=args.use_orthogonal_init
                            )
        
        if self.args.use_rnn: 
            self.rnn = init_rnn(nn.GRUCell(self.hidden_dim, self.hidden_dim), args.use_orthogonal_init)
            self.rnn_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.rnn = MLPBase(self.hidden_dim, self.hidden_dim, 
                               n_hidden_layers=0,
                               use_feature_norm=args.use_obs_norm, 
                               use_orthogonal=args.use_orthogonal_init
                               )

        if self.args.use_popart:
            self.v_out = init_module(PopArt(self.hidden_dim, 1, norm_axes=3, device=args.device), gain=1.0)
        else: 
            self.v_out = init_module(nn.Linear(self.hidden_dim, 1, device=args.device), gain=1.0)
        
        self.value_normalizer = self.v_out if self.args.use_popart else None
        
        obs_dim = scheme["obs"]["vshape"]
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
        
    def init_hidden(self):
        encoder_hidden = self.base.mlp.fc1[0].weight.new(self.args.batch_size, 1, self.n_agents, self.ed_hidden_dim).zero_()
        hidden_base = self.base.mlp.fc1[0].weight.new(self.args.batch_size, 1, self.n_agents, self.hidden_dim).zero_()
        return encoder_hidden, hidden_base 

    def forward(self, batch, hidden_state, t=None, build_inputs=True, vis_mask=None):
        if build_inputs:
            inputs, bs, max_t = self._build_inputs(batch, t=t)
        else: 
            inputs = batch

        orig_batch_dims = inputs.shape[:-1]
        inputs = inputs.reshape(-1, self.input_size)
        
        h_e, h_in = hidden_state
        h_e = h_e.reshape(-1, self.ed_hidden_dim)
        h_in = h_in.reshape(-1, self.hidden_dim)
        
        enc_inputs = build_pairwise_encoder_inputs(self.n_agents, batch, t)
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

        e_i, _, _, _ = self.ifs_architecture(enc_inputs_flat, h_e, flat_mask)

        inputs = th.cat([e_i.detach(), inputs], dim=1)
        x = self.base(inputs)

        if self.args.use_rnn:
            h_out = self.rnn(x, h_in)
            h_norm = self.rnn_norm(h_out)
        else:
            h_norm = h_out = self.rnn(x)

        q = self.v_out(h_norm)
        h_e = h_e.view(*orig_batch_dims, -1)
        h_out = h_out.view(*orig_batch_dims, -1)
        return q.view(*orig_batch_dims, -1), h_e, h_out

    def set_modules(self, ifs_architecture):
        self.ifs_architecture = ifs_architecture

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        inputs.append(batch["obs"][:, ts])
        if self.args.obs_state:
            inputs.append(batch["state"][:, ts].unsqueeze(-2).expand(-1, -1, self.n_agents, -1))
        if self.args.obs_last_action:
            if t is None:
                last_act_all = th.cat([th.zeros_like(batch["actions_onehot"][:, [0]]), 
                                           batch["actions_onehot"][:, slice(0, max_t-1)]],
                                          dim=1)
                inputs.append(last_act_all) 
            elif t == 0: 
                inputs.append(th.zeros_like(batch["actions_onehot"][:, ts]))
            else: 
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)])

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat(inputs, dim=-1)

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_state:
            input_shape += scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents        
        return input_shape
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CommunicationModule(nn.Module):
    def __init__(self, hidden_dim, message_dim):
        super(CommunicationModule, self).__init__()
        self.fc = nn.Linear(hidden_dim, message_dim)
        self.ln = nn.LayerNorm(message_dim)

    def forward(self, b_i):
        return self.ln(F.relu(self.fc(b_i)))


class InformationFusionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(InformationFusionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.eta = nn.Parameter(th.randn(hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, atomic_records, vis_mask=None):
        if atomic_records.dim() == 5:
            bs, max_t, n_agents, n_others, feat_dim = atomic_records.shape
            atomic_records_flat = atomic_records.reshape(bs * max_t * n_agents, n_others, feat_dim)
            mask_flat = vis_mask.reshape(bs * max_t * n_agents, -1) if vis_mask is not None else None
        else:
            total_agents, k, feat_dim = atomic_records.shape
            atomic_records_flat = atomic_records
            mask_flat = vis_mask

        if mask_flat is not None:
            if mask_flat.shape[0] != atomic_records_flat.shape[0]:
                mask_flat = mask_flat[:atomic_records_flat.shape[0]]
            if mask_flat.shape[1] != atomic_records_flat.shape[1]:
                mask_flat = mask_flat[:, :atomic_records_flat.shape[1]]
            mask_3d = mask_flat.unsqueeze(-1).float()
            atomic_records_flat = atomic_records_flat * mask_3d

        phi = F.relu(self.fc1(atomic_records_flat)) # [..., k, hidden_dim]

        scores = th.matmul(phi, self.eta) # [..., k]

        if mask_flat is not None:
            scores = scores.masked_fill(mask_flat == 0, -1e9)

        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # [..., k, 1]

        v_i = th.sum(weights * phi, dim=-2) # [..., hidden_dim]

        e_i = F.relu(self.fc2(v_i + self.eta))
        return e_i


class InformationSeparationModule(nn.Module):
    def __init__(self, fused_dim, hidden_dim, n_train_slots, agent_feature_dim):
        super(InformationSeparationModule, self).__init__()
        self.fc3 = nn.Linear(fused_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_train_slots * agent_feature_dim)
        self.n_train_slots = n_train_slots
        self.agent_feature_dim = agent_feature_dim

    def forward(self, e_i, vis_mask=None):
        bs = e_i.shape[0]
        psi = F.relu(self.fc3(e_i))
        reconstructed = self.fc4(psi)
        reconstructed_slots = reconstructed.view(bs, self.n_train_slots, self.agent_feature_dim)
        
        if vis_mask is not None and vis_mask.shape[-1] == self.n_train_slots:
            mask_expanded = vis_mask.float().unsqueeze(-1)
            reconstructed_slots = reconstructed_slots * mask_expanded
            
        return reconstructed_slots


class IFSArchitectureWithConstraints(nn.Module):
    def __init__(self, args, obs_dim, message_dim, hidden_dim, fused_dim, n_train_slots):
        super(IFSArchitectureWithConstraints, self).__init__()
        self.args = args
        
        self.comm_module = CommunicationModule(hidden_dim, message_dim)
        atomic_input_dim = obs_dim + message_dim
        
        self.ifm = InformationFusionModule(atomic_input_dim, hidden_dim, fused_dim)
        self.ism = InformationSeparationModule(fused_dim, hidden_dim, n_train_slots, atomic_input_dim)
        
        self.state_rnn = nn.GRUCell(fused_dim, hidden_dim)

    def init_hidden(self, batch_size):
        return th.zeros(batch_size, self.state_rnn.hidden_size, device=next(self.parameters()).device)

    def forward(self, atomic_records, hidden_state, vis_mask=None):
        e_i = self.ifm(atomic_records, vis_mask)
        
        b_i = self.state_rnn(e_i, hidden_state)
        
        d_i = self.comm_module(b_i)
        
        reconstructed_data = self.ism(e_i, vis_mask)
        
        return e_i, b_i, d_i, reconstructed_data

def extract_visibility_mask(obs, n_agents, availability_prob=0.7, distance_threshold=None, symmetric=False, occlude_prob=0.0):
    bs, max_t = obs.shape[0], obs.shape[1]
    eye_mask = (1.0 - th.eye(n_agents, device=obs.device)).unsqueeze(0).unsqueeze(0)
    base_mask = eye_mask.expand(bs, max_t, n_agents, n_agents)

    if distance_threshold is not None and obs.shape[-1] >= 2:
        pos = obs[..., :2]
        pos_i = pos.unsqueeze(-2)
        pos_j = pos.unsqueeze(-3)
        dist = th.norm(pos_i - pos_j, dim=-1)
        spatial_mask = (dist <= distance_threshold).float()
        base_mask = base_mask * spatial_mask

    rand_draw = th.rand(bs, max_t, n_agents, n_agents, device=obs.device)
    random_gate = (rand_draw < availability_prob).float()
    
    if symmetric:
        random_gate = random_gate * random_gate.transpose(-1, -2)
    
    visibility_mask = base_mask * random_gate

    if occlude_prob > 0.0:
        occlusion_draw = (th.rand(bs, max_t, n_agents, n_agents, device=obs.device) >= occlude_prob).float()
        visibility_mask = visibility_mask * occlusion_draw

    return visibility_mask


def build_pairwise_encoder_inputs(n_agents, batch, t=None):
    bs = getattr(batch, "batch_size", batch.get("batch_size") if isinstance(batch, dict) else None)
    max_t_val = getattr(batch, "max_seq_length", batch.get("max_seq_length") if isinstance(batch, dict) else None)
    max_t = max_t_val if t is None else 1
    ts = slice(None) if t is None else slice(t, t + 1)

    obs = batch["obs"][:, ts]

    if t is None:
        last_act = th.cat(
            [
                th.zeros_like(batch["actions_onehot"][:, [0]]),
                batch["actions_onehot"][:, slice(0, max_t - 1)],
            ],
            dim=1,
        )
    else:
        last_act = (
            th.zeros_like(batch["actions_onehot"][:, ts])
            if t == 0
            else batch["actions_onehot"][:, slice(t - 1, t)]
        )

    obs_act = th.cat([obs, last_act], dim=-1)
    agent_ids = th.eye(n_agents, device=obs.device)

    inputs_pairwise = []
    for i in range(n_agents):
        others_idx = [j for j in range(n_agents) if j != i]
        tm_obs_act = obs_act[:, :, others_idx]
        self_id = agent_ids[i].view(1, 1, 1, n_agents).expand(bs, max_t, n_agents - 1, n_agents)
        tm_id = agent_ids[others_idx].view(1, 1, n_agents - 1, n_agents).expand(bs, max_t, -1, n_agents)

        pairwise_feat = th.cat([tm_obs_act, self_id, tm_id], dim=-1)
        inputs_pairwise.append(pairwise_feat.unsqueeze(2))

    return th.cat(inputs_pairwise, dim=2)
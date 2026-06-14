import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.smac_obs_parser import SmacObsParser


class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, keys, dist_bias=None):
        b, t, d = keys.shape
        nh = self.n_heads
        hd = self.head_dim

        q = self.q_proj(query).view(b, 1, nh, hd).transpose(1, 2)
        k = self.k_proj(keys).view(b, t, nh, hd).transpose(1, 2)
        v = self.v_proj(keys).view(b, t, nh, hd).transpose(1, 2)

        scores = th.matmul(q, k.transpose(-2, -1)) * self.scale
        if dist_bias is not None:
            scores = scores - dist_bias.view(b, 1, 1, t)

        weights = F.softmax(scores, dim=-1)
        out = th.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(b, 1, d)
        return self.out_proj(out)


class TeacherAttnAgent(nn.Module):
    """CTDS+ teacher: structured kaitu tokens + FP state supplement, cross-attention, GRU, Q output."""

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.hidden_dim = getattr(args, "hidden_dim", args.rnn_hidden_dim)
        self.attn_dim = getattr(args, "attn_dim", self.hidden_dim)
        self.attn_heads = getattr(args, "attn_heads", 4)
        self.use_distance_bias = getattr(args, "use_distance_bias", True)
        self.distance_bias_scale = getattr(args, "distance_bias_scale", 1.0)
        self.use_rnn = getattr(args, "use_rnn", True)

        obs_layout = getattr(args, "obs_layout", None)
        if obs_layout is None:
            raise ValueError(
                "teacher_attn agent requires args.obs_layout from SMAC env_info"
            )
        self.parser = SmacObsParser(obs_layout, self.n_agents)

        self.self_proj = nn.Linear(
            obs_layout["move_feats"] + obs_layout["own_feats"], self.attn_dim
        )
        self.enemy_proj = nn.Linear(obs_layout["n_enemy_feats"], self.attn_dim)
        self.ally_proj = nn.Linear(obs_layout["n_ally_feats"], self.attn_dim)
        self.cooldown_proj = nn.Linear(1, self.attn_dim)

        self.agent_id_emb = nn.Embedding(self.n_agents, self.attn_dim)
        self.hidden_proj = nn.Linear(self.hidden_dim, self.attn_dim)
        self.cross_attn = CrossAttention(self.attn_dim, self.attn_heads)

        if self.use_rnn:
            self.rnn = nn.GRUCell(self.attn_dim, self.hidden_dim)
        else:
            self.rnn = nn.Linear(self.attn_dim, self.hidden_dim)

        self.fc_out = nn.Linear(self.hidden_dim, self.n_actions)

    def init_hidden(self):
        return self.self_proj.weight.new(1, self.hidden_dim).zero_()

    def _build_tokens(self, obs_teacher, state):
        parsed = self.parser.parse_obs(obs_teacher)
        cooldown = self.parser.extract_state_unique(state)

        bs, na = obs_teacher.shape[:2]

        self_tok = self.self_proj(parsed["self_feats"])
        enemy_tok = self.enemy_proj(parsed["enemy_feats"])
        ally_tok = self.ally_proj(parsed["ally_feats"])

        cooldown_exp = cooldown.unsqueeze(1).expand(-1, na, -1, -1)
        global_tok = self.cooldown_proj(cooldown_exp)

        tokens = th.cat(
            [self_tok.unsqueeze(2), enemy_tok, ally_tok, global_tok], dim=2
        )

        n_enemy = parsed["enemy_feats"].shape[2]
        n_ally = parsed["ally_feats"].shape[2]

        dist_bias = th.zeros(bs, na, tokens.shape[2], device=obs_teacher.device)
        if self.use_distance_bias:
            dist_bias[:, :, 1 : 1 + n_enemy] = parsed["enemy_dist"]
            dist_bias[:, :, 1 + n_enemy : 1 + n_enemy + n_ally] = parsed["ally_dist"]
            dist_bias = dist_bias * self.distance_bias_scale

        return tokens, dist_bias

    def forward(self, batch_dict, hidden_state):
        obs_teacher = batch_dict["obs_teacher"]
        state = batch_dict["state"]

        bs, na, _ = obs_teacher.shape
        flat_bs = bs * na

        tokens, dist_bias = self._build_tokens(obs_teacher, state)
        n_tokens = tokens.shape[2]

        tokens_flat = tokens.view(flat_bs, n_tokens, self.attn_dim)
        dist_flat = dist_bias.view(flat_bs, n_tokens)

        agent_ids = (
            th.arange(na, device=obs_teacher.device).unsqueeze(0).expand(bs, -1)
        )
        agent_ids = agent_ids.reshape(flat_bs)
        query = self.agent_id_emb(agent_ids)

        h_in = hidden_state.reshape(flat_bs, self.hidden_dim)
        query = (query + self.hidden_proj(h_in)).unsqueeze(1)

        dist_arg = dist_flat if self.use_distance_bias else None
        attn_out = self.cross_attn(query, tokens_flat, dist_arg).squeeze(1)

        if self.use_rnn:
            h = self.rnn(attn_out, h_in)
        else:
            h = F.relu(self.rnn(attn_out))

        q = self.fc_out(h)
        return q, h.view(bs, na, self.hidden_dim)

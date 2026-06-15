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


class AgentSelfAttention(nn.Module):
    """Self-attention over agent embeddings (inter-agent coordination)."""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = th.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = th.matmul(weights, v)
        return self.out_proj(out)


class TeacherAttnAgent(nn.Module):
    """CTDS+ teacher: unit cross-attn + agent relation + GRU + Q on MAPPO share_obs."""

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
        self.use_agent_relation = getattr(args, "use_agent_relation", True)
        self.agent_rel_layers = max(1, int(getattr(args, "agent_rel_layers", 1)))

        obs_layout = getattr(args, "obs_layout", None)
        if obs_layout is None:
            raise ValueError(
                "teacher_attn agent requires args.obs_layout from SMAC env_info"
            )
        self.parser = SmacObsParser(obs_layout, self.n_agents)
        self.teacher_add_local_obs = obs_layout.get("teacher_add_local_obs", False)

        self.share_self_proj = nn.Linear(
            obs_layout["share_move_feats"] + obs_layout["share_own_feats"],
            self.attn_dim,
        )
        self.share_enemy_proj = nn.Linear(
            obs_layout["share_n_enemy_feats"], self.attn_dim
        )
        self.share_ally_proj = nn.Linear(
            obs_layout["share_n_ally_feats"], self.attn_dim
        )

        if self.teacher_add_local_obs:
            self.local_self_proj = nn.Linear(
                obs_layout["move_feats"] + obs_layout["own_feats"], self.attn_dim
            )
            self.local_enemy_proj = nn.Linear(
                obs_layout["n_enemy_feats"], self.attn_dim
            )
            self.local_ally_proj = nn.Linear(
                obs_layout["n_ally_feats"], self.attn_dim
            )

        self.agent_id_emb = nn.Embedding(self.n_agents, self.attn_dim)
        self.hidden_proj = nn.Linear(self.hidden_dim, self.attn_dim)
        self.cross_attn = CrossAttention(self.attn_dim, self.attn_heads)

        if self.use_agent_relation:
            self.agent_rel = nn.ModuleList(
                [AgentSelfAttention(self.attn_dim) for _ in range(self.agent_rel_layers)]
            )

        if self.use_rnn:
            self.rnn = nn.GRUCell(self.attn_dim, self.hidden_dim)
        else:
            self.rnn = nn.Linear(self.attn_dim, self.hidden_dim)

        self.fc_out = nn.Linear(self.hidden_dim, self.n_actions)

    def init_hidden(self):
        return self.share_self_proj.weight.new(1, self.hidden_dim).zero_()

    def _build_tokens(self, share_obs):
        parsed = self.parser.parse_share_obs(share_obs)
        bs, na = share_obs.shape[:2]

        self_tok = self.share_self_proj(parsed["self_feats"])
        enemy_tok = self.share_enemy_proj(parsed["enemy_feats"])
        ally_tok = self.share_ally_proj(parsed["ally_feats"])
        token_groups = [self_tok.unsqueeze(2), enemy_tok, ally_tok]
        dist_groups = [None, parsed["enemy_dist"], parsed["ally_dist"]]

        if self.teacher_add_local_obs and parsed["local"] is not None:
            local = parsed["local"]
            token_groups.extend(
                [
                    self.local_self_proj(local["self_feats"]).unsqueeze(2),
                    self.local_enemy_proj(local["enemy_feats"]),
                    self.local_ally_proj(local["ally_feats"]),
                ]
            )
            dist_groups.extend([None, local["enemy_dist"], local["ally_dist"]])

        tokens = th.cat(token_groups, dim=2)

        dist_bias = th.zeros(bs, na, tokens.shape[2], device=share_obs.device)
        if self.use_distance_bias:
            offset = 0
            for group_dist in dist_groups:
                if group_dist is None:
                    offset += 1
                    continue
                n_group = group_dist.shape[2]
                dist_bias[:, :, offset : offset + n_group] = group_dist
                offset += n_group
            dist_bias = dist_bias * self.distance_bias_scale

        return tokens, dist_bias

    def _agent_relation(self, x):
        out = x
        for layer in self.agent_rel:
            out = out + layer(out)
        return out

    def forward(self, batch_dict, hidden_state):
        share_obs = batch_dict["share_obs"]

        bs, na, _ = share_obs.shape
        flat_bs = bs * na
        h_in = hidden_state.reshape(flat_bs, self.hidden_dim)

        tokens, dist_bias = self._build_tokens(share_obs)
        n_tokens = tokens.shape[2]

        tokens_flat = tokens.view(flat_bs, n_tokens, self.attn_dim)
        dist_flat = dist_bias.view(flat_bs, n_tokens)

        agent_ids = (
            th.arange(na, device=share_obs.device).unsqueeze(0).expand(bs, -1)
        )
        query = self.agent_id_emb(agent_ids) + self.hidden_proj(
            h_in.view(bs, na, self.hidden_dim)
        )

        unit_out = self.cross_attn(
            query.view(flat_bs, 1, self.attn_dim),
            tokens_flat,
            dist_flat if self.use_distance_bias else None,
        ).squeeze(1)
        local = unit_out.view(bs, na, self.attn_dim)

        if self.use_agent_relation:
            rel = self._agent_relation(local)
            fused = query + local + rel
        else:
            fused = query + local

        if self.use_rnn:
            h = self.rnn(fused.view(flat_bs, self.attn_dim), h_in)
        else:
            h = F.relu(self.rnn(fused.view(flat_bs, self.attn_dim)))

        q = self.fc_out(h)
        return q, h.view(bs, na, self.hidden_dim)

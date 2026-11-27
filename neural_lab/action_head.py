"""
NeuroCraft Scaled-Up Multi-Modal Temporal Action Chunking Transformer (RTX 5060 GPU)
===================================================================================
A ~211-Million-parameter multi-modal embodied policy head that grafts onto the
1,536-dimensional latent representation of Qwen2.5-Coder-1.5B plus grounded 3D world state.

Key Architectural Capabilities:
  1. 85-Token Multi-Modal Context Memory:
     - 4 SLM Intention Tokens projected from 1,536-dim latent h -> [B, 4, 1024]
     - 1 Proprioception Token from 24-dim bodily state -> [B, 1, 1024]
     - 16 Entity Radar Tokens from up to 16 nearby entities -> [B, 16, 1024]
     - 64 Semantic 3D Voxel Patch Tokens (15x15x15 32-class block IDs -> 4x4x4 grid)
  2. Unified Physical + Tool-Dispatch Outputs across 8 Future Ticks (400ms @ 20Hz):
     - Continuous Camera Aim: (delta_yaw, delta_pitch)
     - Continuous Planar Velocity: (vx, vz)
     - Discrete Motor & Inventory Actions: [jump, sprint, sneak, attack, use_item, swap_slot, dig_block, drop_item]
     - Discrete Tool Dispatch Head (24 canonical MCP/motor tools + invocation gate)
     - Sub-Goal Lifecycle Head: [subgoal_completed, replan_needed] & Critic V(s)
"""

import math
from typing import Dict, Optional, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

PROPRIOCEPTION_DIM = 24  # hp, food, yaw, pitch, vx, vy, vz, on_ground, in_water, target_dx/dy/dz/dist, fuse, items
LATENT_DIM = 1536        # Qwen2.5-Coder-1.5B hidden_size
HIDDEN_DIM = 1024        # Scaled up transformer hidden dimension (~211M params)
NUM_LAYERS = 12          # 12 Cross-Attention Transformer Decoder layers
NUM_HEADS = 16           # 16 Attention Heads (64 dims per head)
CHUNK_SIZE = 8           # 8 future ticks (400ms at 20Hz)
VOXEL_VOCAB_SIZE = 32    # 32 semantic block categories (0=air..31=interactive)
NUM_ENTITIES = 16        # Top 16 nearest entities in 360-degree radar
ENTITY_FEAT_DIM = 12     # Per-entity spatial, velocity, type, and fuse features
AIM_STD_MIN = 3.5        # Minimum continuous aim std (deg) to prevent mode collapse
AIM_STD_MAX = 45.0
MOVE_STD_MIN = 0.08
MOVE_STD_MAX = 0.65

CANONICAL_MOTOR_TOOLS: List[str] = [
    "none",
    "craft_item",
    "smelt_item",
    "equip_armor_or_item",
    "store_items_in_chest",
    "loot_chest",
    "eat_food",
    "give_item_to_player",
    "gather_resource",
    "explore_or_dig_for_ores",
    "engage_combat",
    "shoot_bow",
    "navigate_parkour_or_bridge",
    "navigate_to_coordinates",
    "follow_player",
    "come_here",
    "pillar_up",
    "construct_modular_structure",
    "till_and_farm",
    "tame_animal",
    "breed_animals",
    "shear_sheep",
    "use_tool_on_block",
    "light_block_fire",
]


class MinecraftActionHead(nn.Module):
    """
    Scaled-Up Multi-Modal Vision-Language-Action (VLA) Temporal Action Transformer.
    ~211M parameters (~422 MB in BF16), <3ms chunk inference on RTX 5060.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        proprioception_dim: int = PROPRIOCEPTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        chunk_size: int = CHUNK_SIZE,
        voxel_vocab_size: int = VOXEL_VOCAB_SIZE,
        num_entities: int = NUM_ENTITIES,
        entity_feat_dim: int = ENTITY_FEAT_DIM,
        num_tools: int = len(CANONICAL_MOTOR_TOOLS),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.proprioception_dim = proprioception_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.voxel_vocab_size = voxel_vocab_size
        self.num_entities = num_entities
        self.entity_feat_dim = entity_feat_dim
        self.num_tools = num_tools

        # 1. SLM Latent Intention Encoders -> 4 Context Tokens [B, 4, hidden_dim]
        self.proj_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 4),
            nn.GELU(),
        )
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.latent_pos_emb = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)

        # 2. Bodily Proprioception Encoder -> 1 Context Token [B, 1, hidden_dim]
        self.proj_proprio = nn.Sequential(
            nn.Linear(proprioception_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3. Entity Radar Encoder -> 16 Context Tokens [B, 16, hidden_dim]
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.entity_pos_emb = nn.Parameter(torch.randn(1, num_entities, hidden_dim) * 0.02)

        # 4. Semantic 3D Voxel Perception Branch (15x15x15 -> 4x4x4 = 64 Spatial Tokens)
        self.voxel_embed = nn.Embedding(voxel_vocab_size, 32)
        self.voxel_conv3d = nn.Sequential(
            nn.Conv3d(32, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
        )
        self.voxel_patch_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.voxel_pos_emb = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)

        # 5. Learned Trajectory Query Embeddings (for K-tick action chunking)
        self.learned_query = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)

        # 6. Temporal Cross-Attention Transformer Decoder Backbone (12 Layers, 16 Heads)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            dropout=0.05,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 7. Trajectory Prediction Heads across the K-tick horizon
        # 7a. Continuous Camera Aim Trajectory: (delta_yaw, delta_pitch) in [-180, 180] deg
        self.aim_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

        # 7b. Continuous Movement Trajectory: (vx, vz) in [-1.0, 1.0]
        self.move_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

        # 7c. Discrete Core Motor Logits across 8 ticks: [jump, sprint, sneak, attack, use_item, swap_slot]
        self.discrete_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 6),
        )

        # 7d. Extended Motor/Inventory Logits across 8 ticks: [dig_block, drop_item]
        self.extended_motor_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )

        # 7e. Tool Dispatch & Invocation Gate Heads (from pooled chunk representation)
        self.tool_dispatch_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_tools),
        )
        self.tool_gate_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # 7f. Sub-Goal Lifecycle Head: [subgoal_completed_logit, replan_needed_logit]
        self.subgoal_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        # 7g. State Value Critic V(s) baseline for PPO / GRPO advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        # 7h. Learnable continuous PPO exploration stds (aim [deg], move [norm])
        self.aim_log_std = nn.Parameter(torch.tensor([math.log(12.0), math.log(6.0)], dtype=torch.float32))
        self.move_log_std = nn.Parameter(torch.tensor([math.log(0.25), math.log(0.25)], dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small output init to prevent initial erratic motor spikes
        nn.init.orthogonal_(self.aim_head[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.move_head[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.discrete_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.extended_motor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.tool_dispatch_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.tool_gate_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.subgoal_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(
        self,
        latent_h: torch.Tensor,
        proprioception: torch.Tensor,
        voxels: Optional[torch.Tensor] = None,
        entities: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass from SLM latent intention + bodily proprioception + 3D semantic voxels + entity radar
        to an 8-tick continuous motor trajectory chunk and discrete tool/lifecycle predictions.
        """
        if latent_h.dim() == 1:
            latent_h = latent_h.unsqueeze(0)
        if proprioception.dim() == 1:
            proprioception = proprioception.unsqueeze(0)

        p = next(self.parameters())
        latent_h = latent_h.to(device=p.device, dtype=p.dtype)
        proprioception = proprioception.to(device=p.device, dtype=p.dtype)

        B = latent_h.shape[0]

        # 1. SLM Latent Intent Tokens [B, 4, hidden_dim]
        h_proj = self.proj_latent(latent_h).view(B, 4, self.hidden_dim)
        h_tokens = self.latent_norm(h_proj) + self.latent_pos_emb  # [B, 4, hidden_dim]

        # 2. Bodily Proprioception Token [B, 1, hidden_dim]
        p_tok = self.proj_proprio(proprioception).unsqueeze(1)      # [B, 1, hidden_dim]

        # 3. Entity Radar Tokens [B, 16, hidden_dim]
        if entities is None:
            entities = torch.zeros((B, self.num_entities, self.entity_feat_dim), device=p.device, dtype=p.dtype)
        else:
            if entities.dim() == 2:
                entities = entities.unsqueeze(0)
            entities = entities.to(device=p.device, dtype=p.dtype)
        e_tokens = self.entity_encoder(entities) + self.entity_pos_emb  # [B, 16, hidden_dim]

        # 4. Spatial 3D Semantic Voxel Tokens [B, 64, hidden_dim]
        if voxels is None:
            voxel_ids = torch.zeros((B, 15, 15, 15), device=p.device, dtype=torch.long)
        else:
            if voxels.dim() == 3:
                voxels = voxels.unsqueeze(0)
            elif voxels.dim() == 5:
                voxels = voxels.squeeze(1)
            voxel_ids = voxels.long().clamp(0, self.voxel_vocab_size - 1).to(device=p.device)

        # Embed [B, 15, 15, 15] -> [B, 15, 15, 15, 32] -> [B, 32, 15, 15, 15]
        vox_emb = self.voxel_embed(voxel_ids).permute(0, 4, 1, 2, 3).contiguous().to(dtype=p.dtype)
        vox_feat = self.voxel_conv3d(vox_emb)                      # [B, 256, 4, 4, 4]
        vox_patches = vox_feat.permute(0, 2, 3, 4, 1).contiguous().view(B, 64, 256)
        v_tokens = self.voxel_patch_proj(vox_patches) + self.voxel_pos_emb  # [B, 64, hidden_dim]

        # Combine into 85-Token Multi-Modal Context Memory [B, 85, hidden_dim]
        context = torch.cat([h_tokens, p_tok, e_tokens, v_tokens], dim=1)

        # Expand learned trajectory queries across batch
        tgt = self.learned_query.unsqueeze(0).expand(B, -1, -1)    # [B, 8, hidden_dim]

        # Decode action chunk through 12-layer cross-attention transformer
        decoded = self.decoder(tgt=tgt, memory=context)            # [B, 8, hidden_dim]
        pooled = decoded.mean(dim=1)                               # [B, hidden_dim]

        # Trajectory outputs
        delta_angles = self.aim_head(decoded) * 180.0              # [B, 8, 2]
        movement = self.move_head(decoded)                         # [B, 8, 2]
        discrete_logits = self.discrete_head(decoded)              # [B, 8, 6]
        extended_logits = self.extended_motor_head(decoded)        # [B, 8, 2]

        # Pooled decision outputs
        tool_logits = self.tool_dispatch_head(pooled)              # [B, num_tools]
        tool_gate_logit = self.tool_gate_head(pooled).squeeze(-1)  # [B]
        subgoal_logits = self.subgoal_head(pooled)                 # [B, 2]
        value = self.value_head(decoded[:, 0, :]).squeeze(-1)      # [B]

        return {
            "delta_yaw": delta_angles[:, 0, 0],
            "delta_pitch": delta_angles[:, 0, 1],
            "vx": movement[:, 0, 0],
            "vz": movement[:, 0, 1],
            "discrete_logits": discrete_logits[:, 0, :],
            "discrete_probs": torch.sigmoid(discrete_logits[:, 0, :]),
            "value": value,
            # Full 8-tick temporal trajectory chunk
            "trajectory_aim": delta_angles,
            "trajectory_movement": movement,
            "trajectory_discrete_logits": discrete_logits,
            "trajectory_discrete_probs": torch.sigmoid(discrete_logits),
            "trajectory_extended_logits": extended_logits,
            "trajectory_extended_probs": torch.sigmoid(extended_logits),
            # High-level tool dispatch & sub-goal lifecycle signals
            "tool_logits": tool_logits,
            "tool_probs": F.softmax(tool_logits, dim=-1),
            "tool_gate_logit": tool_gate_logit,
            "tool_gate_prob": torch.sigmoid(tool_gate_logit),
            "subgoal_logits": subgoal_logits,
            "subgoal_probs": torch.sigmoid(subgoal_logits),
        }

    def get_action_stds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns bounded float32 continuous Gaussian std tensors ([2] aim deg, [2] move)."""
        aim_std = self.aim_log_std.float().clamp(math.log(AIM_STD_MIN), math.log(AIM_STD_MAX)).exp()
        move_std = self.move_log_std.float().clamp(math.log(MOVE_STD_MIN), math.log(MOVE_STD_MAX)).exp()
        return aim_std, move_std

    def evaluate_action_chunk(
        self,
        out: Dict[str, torch.Tensor],
        aim_actions: torch.Tensor,
        move_actions: torch.Tensor,
        disc_actions: torch.Tensor,
        ext_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates joint Gaussian+Bernoulli log-probability, scale-balanced entropy,
        and Critic V(s) in float32 for clipped PPO updates.
        """
        aim_mu = out["trajectory_aim"].float()
        move_mu = out["trajectory_movement"].float()
        aim_std, move_std = self.get_action_stds()

        aim_dist = torch.distributions.Normal(aim_mu, aim_std.view(1, 1, 2))
        move_dist = torch.distributions.Normal(move_mu, move_std.view(1, 1, 2))
        disc_dist = torch.distributions.Bernoulli(
            logits=out["trajectory_discrete_logits"].float().clamp(-10.0, 10.0)
        )
        ext_dist = torch.distributions.Bernoulli(
            logits=out["trajectory_extended_logits"].float().clamp(-10.0, 10.0)
        )

        lp_aim = aim_dist.log_prob(aim_actions.float()).mean(dim=(1, 2))
        lp_move = move_dist.log_prob(move_actions.float()).mean(dim=(1, 2))
        lp_disc = disc_dist.log_prob(disc_actions.float().clamp(0.0, 1.0)).mean(dim=(1, 2))
        lp_ext = ext_dist.log_prob(ext_actions.float().clamp(0.0, 1.0)).mean(dim=(1, 2))
        joint_log_prob = 0.35 * lp_aim + 0.35 * lp_move + 0.25 * lp_disc + 0.05 * lp_ext

        ent_aim = (0.5 * (1.0 + math.log(2.0 * math.pi)) + torch.log((aim_std / 180.0).clamp(min=1e-4))).mean()
        ent_move = move_dist.entropy().mean(dim=(1, 2))
        ent_disc = disc_dist.entropy().mean(dim=(1, 2))
        ent_ext = ext_dist.entropy().mean(dim=(1, 2))
        joint_entropy = 0.25 * ent_aim + 0.25 * ent_move + 0.40 * ent_disc + 0.10 * ent_ext

        return joint_log_prob, joint_entropy, out["value"].float()

    def predict_action_dict(
        self,
        latent_h: torch.Tensor,
        proprioception: torch.Tensor,
        voxels: Optional[torch.Tensor] = None,
        entities: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        explore: bool = False,
        aim_noise_std: Optional[float] = None,
        move_noise_std: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Inference helper returning clean Python primitives ready for Mineflayer bot,
        including tick-0 controls, full 8-tick action chunk, and recommended MCP tool dispatch.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(latent_h, proprioception, voxels=voxels, entities=entities)
            traj_aim_t = torch.nan_to_num(out["trajectory_aim"][0].float(), nan=0.0, posinf=0.0, neginf=0.0)
            traj_move_t = torch.nan_to_num(out["trajectory_movement"][0].float(), nan=0.0, posinf=0.0, neginf=0.0)
            traj_probs_t = torch.nan_to_num(out["trajectory_discrete_probs"][0].float(), nan=0.5, posinf=1.0, neginf=0.0)
            ext_probs_t = torch.nan_to_num(out["trajectory_extended_probs"][0].float(), nan=0.0, posinf=1.0, neginf=0.0)

            if explore:
                learned_aim_std, learned_move_std = self.get_action_stds()
                eff_aim_std = aim_noise_std if aim_noise_std is not None else learned_aim_std.to(traj_aim_t.device).unsqueeze(0)
                eff_move_std = move_noise_std if move_noise_std is not None else learned_move_std.to(traj_move_t.device).unsqueeze(0)
                traj_aim_t = traj_aim_t + torch.randn_like(traj_aim_t) * eff_aim_std
                traj_aim_t[:, 0] = traj_aim_t[:, 0].clamp(-180.0, 180.0)
                traj_aim_t[:, 1] = traj_aim_t[:, 1].clamp(-90.0, 90.0)
                traj_move_t = (traj_move_t + torch.randn_like(traj_move_t) * eff_move_std).clamp(-1.0, 1.0)
                disc_sampled = torch.bernoulli(traj_probs_t.clamp(0.02, 0.98)) >= 0.5
                ext_sampled = torch.bernoulli(ext_probs_t.clamp(0.02, 0.98)) >= 0.5
            else:
                disc_sampled = traj_probs_t >= threshold
                ext_sampled = ext_probs_t >= threshold

            traj_aim = traj_aim_t.tolist()
            traj_move = traj_move_t.tolist()
            disc_bools = disc_sampled.tolist()
            ext_bools = ext_sampled.tolist()

            action_chunk: List[Dict[str, Any]] = []
            for k in range(self.chunk_size):
                d_k = disc_bools[k]
                e_k = ext_bools[k]
                action_chunk.append({
                    "tick": k,
                    "delta_yaw": round(traj_aim[k][0], 2),
                    "delta_pitch": round(traj_aim[k][1], 2),
                    "vx": round(traj_move[k][0], 3),
                    "vz": round(traj_move[k][1], 3),
                    "jump": bool(d_k[0]),
                    "sprint": bool(d_k[1]),
                    "sneak": bool(d_k[2]),
                    "attack": bool(d_k[3]),
                    "use_item": bool(d_k[4]),
                    "swap_slot": bool(d_k[5]),
                    "dig_block": bool(e_k[0]),
                    "drop_item": bool(e_k[1]),
                })

            tool_probs = torch.nan_to_num(out["tool_probs"][0].float(), nan=0.0, posinf=0.0, neginf=0.0)
            top_tool_idx = int(torch.argmax(tool_probs).item())
            top_tool_name = CANONICAL_MOTOR_TOOLS[top_tool_idx] if top_tool_idx < len(CANONICAL_MOTOR_TOOLS) else "none"
            top_tool_conf = float(tool_probs[top_tool_idx].item())
            tool_gate_prob = float(torch.nan_to_num(out["tool_gate_prob"][0].float(), nan=0.0).item())
            subgoal_probs = torch.nan_to_num(out["subgoal_probs"][0].float(), nan=0.0).tolist()
            state_value = float(torch.nan_to_num(out["value"][0].float(), nan=0.0).item())
            t0_chunk = action_chunk[0]

            return {
                "delta_yaw": t0_chunk["delta_yaw"],
                "delta_pitch": t0_chunk["delta_pitch"],
                "vx": t0_chunk["vx"],
                "vz": t0_chunk["vz"],
                "jump": t0_chunk["jump"],
                "sprint": t0_chunk["sprint"],
                "sneak": t0_chunk["sneak"],
                "attack": t0_chunk["attack"],
                "use_item": t0_chunk["use_item"],
                "swap_slot": t0_chunk["swap_slot"],
                "dig_block": t0_chunk["dig_block"],
                "drop_item": t0_chunk["drop_item"],
                "recommended_tool": top_tool_name,
                "recommended_tool_confidence": round(top_tool_conf, 3),
                "should_invoke_tool": bool(tool_gate_prob >= threshold and top_tool_name != "none"),
                "subgoal_completed": bool(subgoal_probs[0] >= threshold),
                "should_replan": bool(subgoal_probs[1] >= threshold),
                "state_value": round(state_value, 3),
                "action_chunk": action_chunk,
            }

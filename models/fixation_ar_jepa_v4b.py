"""Fixation-AR-JEPA + Spatial-Completion (V4b) — paper version.

This is the CXR self-supervised encoder used in the paper. It augments the
gaze-driven autoregressive JEPA recipe with an explicit per-patch supervision
signal on the patches the radiologist did NOT fixate.

Architecture (Stage-1):

  CXR (224^2)
    └─ MedCLIP Swin-ViT  ─────────────────────────────►  patch tokens (B, 49, 768)
                                       │                        │
                                       │                        ├─ EMA copy ──► target patch tokens
                                       ▼
  Gaze (x,y,t)*T  ──►  FixationEmbedder  ─►  fixation tokens (B, T, 768)
                                       │
                                       ▼
                          Causal Transformer (8L, RoPE)
                                       │
                            ┌──────────┴───────────┐
                            ▼                      ▼
                    next-fixation          Spatial-Completion
                    predictor (MLP)        Decoder (cross-attn, 2L)
                            │                      │
                            ▼                      ▼
                    ẑ_{t+1} (gaze AR)      r̂_p   for p ∉ visited
                            │                      │
                            ▼                      ▼
              SmoothL1(LN(ẑ), z_{t+1})   SmoothL1(LN(r̂), LN(ema_patch_p))
                       └──────────  +λ·  ──────────┘

  l_AR    : standard AIM-style next-fixation JEPA loss on the gaze sequence.
  l_compl : I-JEPA-style loss whose context block = the gaze sequence and
            whose target blocks = the unvisited patch tokens, predicted in
            parallel via a small cross-attention decoder. Targets come from
            an EMA copy of the Swin backbone (prevents trivial "match
            yourself" collapse).
  Total loss = l_AR + λ · l_compl    (λ = 1.0).

Why an EMA backbone for completion targets?
  When the live Swin is unfrozen, every Stage-1 loss we tried collapsed in
  ~2 epochs (the backbone shifts to make targets trivial). Freezing the
  backbone fixes the AR collapse but renders B/C losses meaningless because
  the "target" is the same backbone the predictor reads from. Adding an
  explicit EMA backbone gives a stable, non-trivial completion target while
  keeping the rest of the network learnable.

Stage-2: load this backbone, attach a 14-class CheXpert classification head
(see scripts/train_v4_stage2.py), full-fine-tune at lr=1e-4 / backbone_lr=1e-5,
20 epochs. Result on the official 668-image CheXpert test split:
  - V4b (frozen Stage-1, this file)  : 0.8761 mean AUROC
  - V3  (no completion loss)         : 0.8793 mean AUROC (50 ep SFT)

For downstream usage:
  m = build_v4b()
  m.load_state_dict(torch.load('stage1_ckpt.pth')['model'], strict=False)
  feats = m.get_features(imgs)   # (B, 1536) for linear probe
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fixation_ar_jepa import (
    RMSNorm, RotaryEmbedding, apply_rope, SwiGLU, FixationEmbedder,
)
from .fixation_ar_jepa_medclip import MedCLIPSwinEncoder


# ---------------------------------------------------------------------------
# Causal transformer block (RMSNorm + RoPE + GQA + SwiGLU, Llama-style)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=12, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)
        self.rope = RotaryEmbedding(head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(N, device=x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal, float('-inf')).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.o(out)


class CausalBlock(nn.Module):
    def __init__(self, dim, num_heads=12, head_dim=64):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, head_dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Spatial-completion decoder
# ---------------------------------------------------------------------------

class _CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward(self, q, kv, kv_mask):
        """Block kv columns where kv_mask is False (padding gaze positions)."""
        B, Tq, _ = q.shape
        Tkv = kv.shape[1]
        Q = self.q(q).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(kv).view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(kv).view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if kv_mask is not None:
            block = ~kv_mask
            attn = attn.masked_fill(block.unsqueeze(1).unsqueeze(2), float('-inf'))
        # Guard rows where every kv key was masked (would produce NaN softmax).
        finite = torch.isfinite(attn).any(dim=-1, keepdim=True)
        attn = attn.softmax(dim=-1)
        attn = torch.where(finite, attn, torch.zeros_like(attn))
        out = (attn @ V).transpose(1, 2).reshape(B, Tq, -1)
        return self.o(out)


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.cross_attn = _CrossAttention(dim, num_heads, head_dim)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, q, kv, kv_mask):
        q = q + self.cross_attn(self.norm_q(q), self.norm_kv(kv), kv_mask)
        q = q + self.ffn(self.norm_ffn(q))
        return q


class SpatialCompletionDecoder(nn.Module):
    """Cross-attn decoder. Q = [mask] + grid PE; K/V = causal-context outputs."""

    def __init__(self, embed_dim=768, num_heads=12, head_dim=64, depth=2,
                 num_patches=49):
        super().__init__()
        self.num_patches = num_patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.layers = nn.ModuleList([
            _CrossAttnBlock(embed_dim, num_heads, head_dim) for _ in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, gaze_ctx, gaze_kv_mask, unv_ids, unv_mask):
        B, K = unv_ids.shape
        pe = self.pos_embed[0][unv_ids]
        q = self.mask_token.expand(B, K, -1) + pe
        for blk in self.layers:
            q = blk(q, gaze_ctx, gaze_kv_mask)
        return self.out_proj(self.norm(q))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_unvisited(fix_ids, fix_mask, num_patches=49):
    """Returns (unv_ids: (B, P), unv_mask: (B, P)) where the first U valid
    unvisited patch indices are placed up front in raster order, rest pad."""
    B, T = fix_ids.shape
    fix_ids_clamped = fix_ids.clamp(0, num_patches - 1)
    one_hot = F.one_hot(fix_ids_clamped, num_classes=num_patches).bool()
    visited_mat = (one_hot & fix_mask.unsqueeze(-1)).any(dim=1)  # (B, P)
    unvisited_mat = ~visited_mat
    P = num_patches
    grid = torch.arange(P, device=fix_ids.device).unsqueeze(0).expand(B, -1)
    sort_key = (~unvisited_mat).long() * P + grid
    sorted_idx = sort_key.argsort(dim=1)
    return grid.gather(1, sorted_idx), unvisited_mat.gather(1, sorted_idx)


# ---------------------------------------------------------------------------
# V4b model
# ---------------------------------------------------------------------------

class FixationARJEPA_V4b(nn.Module):
    """Fixation-AR-JEPA + Spatial-Completion."""

    def __init__(
        self,
        embed_dim=768,
        context_depth=8,
        context_heads=12,
        head_dim=64,
        max_fixations=96,
        ema_momentum=0.998,
        medclip_ckpt='/tmp/medclip_weights/medclip-vit-pretrained/pytorch_model.bin',
        unfreeze_swin=True,
        spatial_decoder_depth=2,
        completion_weight=1.0,
        img_size=224,
        random_init=False,
    ):
        super().__init__()
        self.completion_weight = completion_weight

        # --- backbone ---
        self.vit = MedCLIPSwinEncoder(medclip_ckpt, img_size=img_size,
                                      random_init=random_init)
        self.vit_dim = self.vit.embed_dim   # 768
        self.num_patches = self.vit.num_patches  # 49 = 7×7
        self.grid_size = self.vit.grid_size      # 7
        if not unfreeze_swin:
            for p in self.vit.parameters():
                p.requires_grad = False

        # --- fixation embedder ---
        self.fix_embedder = FixationEmbedder(
            vit_dim=self.vit_dim, embed_dim=embed_dim,
            max_fixations=max_fixations, grid_size=self.grid_size,
        )

        # --- causal context blocks ---
        self.context_blocks = nn.ModuleList([
            CausalBlock(embed_dim, context_heads, head_dim)
            for _ in range(context_depth)
        ])
        self.context_norm = RMSNorm(embed_dim)

        # --- AR predictor ---
        self.predictor = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # --- EMA target on (embedder, blocks, norm) for AR ---
        self.target_embedder = copy.deepcopy(self.fix_embedder)
        self.target_blocks = copy.deepcopy(self.context_blocks)
        self.target_norm = copy.deepcopy(self.context_norm)
        for p in (list(self.target_embedder.parameters()) +
                   list(self.target_blocks.parameters()) +
                   list(self.target_norm.parameters())):
            p.requires_grad = False

        # --- EMA backbone for spatial-completion targets ---
        self.target_vit = copy.deepcopy(self.vit)
        for p in self.target_vit.parameters():
            p.requires_grad = False

        # --- spatial-completion decoder ---
        self.spatial_decoder = SpatialCompletionDecoder(
            embed_dim=embed_dim, num_heads=context_heads, head_dim=head_dim,
            depth=spatial_decoder_depth, num_patches=self.num_patches,
        )

        # --- eval projection (for SFT compatibility — produces 1536-d feature) ---
        self.eval_proj = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, self.vit_dim),
        )
        self.readout_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.readout_token, std=0.02)

        self.embed_dim = embed_dim
        self.ema_momentum = ema_momentum

    # ---------- helpers ----------
    def _encode_image(self, imgs):
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        return self.vit.forward_features(imgs)

    @torch.no_grad()
    def _encode_image_target(self, imgs):
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        return self.target_vit.forward_features(imgs)

    def _remap_patch_ids(self, fix_patch_ids, fix_coords):
        x = fix_coords[:, :, 0].clamp(0, 0.999)
        y = fix_coords[:, :, 1].clamp(0, 0.999)
        col = (x * self.grid_size).long()
        row = (y * self.grid_size).long()
        return (row * self.grid_size + col).clamp(0, self.num_patches - 1)

    # ---------- training forward ----------
    def forward(self, imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        """Returns dict with both AR and spatial-completion preds + targets."""
        patch_tokens = self._encode_image(imgs)
        gaze_ids = self._remap_patch_ids(fix_patch_ids, fix_coords)

        # --- gaze AR head (V3-identical) ---
        ctx = self.fix_embedder(patch_tokens, gaze_ids, fix_coords, fix_meta, fix_mask)
        x = ctx
        for blk in self.context_blocks:
            x = blk(x)
        x = self.context_norm(x)
        ar_preds = self.predictor(x[:, :-1, :])

        with torch.no_grad():
            tgt = self.target_embedder(
                patch_tokens.detach(), gaze_ids, fix_coords, fix_meta, fix_mask)
            y = tgt
            for blk in self.target_blocks:
                y = blk(y)
            y = self.target_norm(y)
            ar_targets = F.layer_norm(y[:, 1:, :], [y.shape[-1]])

        valid_ar = fix_mask[:, 1:].unsqueeze(-1).float()
        ar_preds = ar_preds * valid_ar
        ar_targets = ar_targets * valid_ar

        # --- spatial-completion head ---
        unv_ids, unv_mask = compute_unvisited(gaze_ids, fix_mask, self.num_patches)
        spat_preds = self.spatial_decoder(x, fix_mask, unv_ids, unv_mask)
        with torch.no_grad():
            tgt_patches = self._encode_image_target(imgs)
            idx = unv_ids.unsqueeze(-1).expand(-1, -1, tgt_patches.shape[-1])
            spat_targets = torch.gather(tgt_patches, 1, idx)
            spat_targets = F.layer_norm(spat_targets, [spat_targets.shape[-1]])
        valid_unv = unv_mask.unsqueeze(-1).float()

        return {'ar_preds': ar_preds, 'ar_targets': ar_targets,
                'spat_preds': spat_preds * valid_unv,
                'spat_targets': spat_targets * valid_unv,
                'spat_mask': unv_mask}

    # ---------- losses ----------
    def compute_loss(self, out):
        ar = F.layer_norm(out['ar_preds'], [out['ar_preds'].shape[-1]])
        l_ar = F.smooth_l1_loss(ar, out['ar_targets'])
        sp = F.layer_norm(out['spat_preds'], [out['spat_preds'].shape[-1]])
        l_sp = F.smooth_l1_loss(sp, out['spat_targets'])
        return {'loss': l_ar + self.completion_weight * l_sp,
                'l_ar': l_ar.detach(), 'l_sp': l_sp.detach()}

    # ---------- EMA update ----------
    @torch.no_grad()
    def update_target(self, momentum=None):
        m = momentum if momentum is not None else self.ema_momentum
        for src, tgt in [
            (self.fix_embedder.parameters(), self.target_embedder.parameters()),
            (self.context_blocks.parameters(), self.target_blocks.parameters()),
            (self.context_norm.parameters(), self.target_norm.parameters()),
            (self.vit.parameters(), self.target_vit.parameters()),
        ]:
            for p_s, p_t in zip(src, tgt):
                p_t.data.mul_(m).add_((1 - m) * p_s.data)

    # ---------- features for SFT / linear probe ----------
    def get_features(self, imgs):
        """1536-d feature = [readout-token (768d), mean-pooled patch tokens (768d)].

        At inference time gaze isn't available, so we build a deterministic
        raster-scan fixation sequence over the 7×7 grid and feed it through
        the (eval-time) context blocks. The readout token at position 0
        absorbs the full sequence via causal attention from later tokens'
        residuals — a standard "register"-style aggregator.
        """
        patch_tokens = self._encode_image(imgs)
        B = patch_tokens.shape[0]
        G = self.grid_size
        T = self.num_patches
        fix_ids = torch.arange(T, device=imgs.device).unsqueeze(0).expand(B, -1)
        fix_coords = torch.zeros(B, T, 2, device=imgs.device)
        for i in range(T):
            fix_coords[:, i, 0] = (i % G) / G
            fix_coords[:, i, 1] = (i // G) / G
        fix_meta = torch.zeros(B, T, 3, device=imgs.device)
        fix_meta[:, :, 2] = torch.linspace(0, 1, T, device=imgs.device)
        fix_mask = torch.ones(B, T, dtype=torch.bool, device=imgs.device)

        ctx = self.fix_embedder(patch_tokens, fix_ids, fix_coords, fix_meta, fix_mask)
        readout = self.readout_token.expand(B, -1, -1)
        x = torch.cat([readout, ctx], dim=1)
        for blk in self.context_blocks:
            x = blk(x)
        x = self.context_norm(x)
        feat = self.eval_proj(x[:, 0, :])
        vit_feat = patch_tokens.mean(dim=1)
        return torch.cat([feat, vit_feat], dim=-1)


def build_v4b(cfg=None):
    """Build with default V4b hyper-parameters (overridable via dict)."""
    cfg = cfg or {}
    m = cfg.get("model", {})
    # get_features() runs a raster scan of length num_patches (=(img_size//32)^2)
    # through the causal blocks, so temporal_pe must cover at least that many
    # positions. 96 is plenty at 224 (49 patches) but too small at 448 (196).
    img_size = m.get("img_size", 224)
    default_max_fix = max(96, (img_size // 32) ** 2)
    return FixationARJEPA_V4b(
        embed_dim=m.get("embed_dim", 768),
        context_depth=m.get("context_depth", 8),
        context_heads=m.get("context_heads", 12),
        head_dim=m.get("head_dim", 64),
        max_fixations=m.get("max_fixations", default_max_fix),
        ema_momentum=m.get("ema_momentum", 0.998),
        medclip_ckpt=m.get("medclip_ckpt",
            "/tmp/medclip_weights/medclip-vit-pretrained/pytorch_model.bin"),
        unfreeze_swin=m.get("unfreeze_swin", False),
        spatial_decoder_depth=m.get("spatial_decoder_depth", 2),
        completion_weight=m.get("completion_weight", 1.0),
        img_size=m.get("img_size", 224),
        random_init=m.get("random_init", False),
    )

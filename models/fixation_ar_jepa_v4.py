"""V4: Fixation-AR-JEPA variants that put unvisited patches to work.

Three modes, all built on V3's MedCLIP-Swin + FixationEmbedder + 8-layer
causal transformer + EMA(embedder/blocks/norm) + smooth-L1 loss:

  mode='a' Extended-AR
    Sequence = [valid gaze tokens, valid unvisited tokens (raster order), pad].
    Single causal mask. AR loss covers all 49 patches.

  mode='b' AR + Spatial-Completion
    AR loss on gaze (V3-identical). NEW: small cross-attention decoder
    queries learnable [mask] + per-grid PE for unvisited positions, K/V =
    causal-block outputs over gaze. Targets = EMA-Swin patch_tokens at
    those positions (LayerNorm + smooth-L1). Adds an EMA Swin backbone.

  mode='c' AR + Masked-Completion (single sequence)
    Append learnable mask tokens to the sequence after gaze. Custom 2-D
    attention mask: gaze causal among themselves; mask tokens attend to
    valid gaze tokens + their own diagonal (parallel reconstruction).
    Targets = LIVE Swin patch_tokens at unvisited positions (stop-grad +
    LayerNorm). No EMA Swin needed (mask tokens never see their target).

  mode='d' Unvisited-prefix → gaze AR (cognitive setup)
    Sequence = [unvisited tokens (patch features), gaze tokens]. Prefix-LM
    attention: unvisited tokens attend bidirectionally to themselves; gaze
    tokens attend causally to (full prefix + earlier gaze). AR loss only on
    gaze. Models the radiologist process: see whole image → predict gaze
    trajectory. No new heads; unvisited patches contribute via implicit
    supervision (be useful for predicting gaze).

Loss = l_ar + completion_weight * l_completion (modes b, c only).
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
# Maskable self-attention (replaces V3's hardcoded-causal block)
# ---------------------------------------------------------------------------

class MaskableSelfAttention(nn.Module):
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

    def forward(self, x, attn_mask=None):
        """attn_mask: None → causal; (N, N) bool or (B, N, N) bool, True = block."""
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(N, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        if attn_mask.dim() == 2:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        else:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        # Guard fully-masked rows (all -inf → softmax NaN). Replace -inf with 0
        # for those rows after softmax (the resulting row will sum to 0 and the
        # downstream zero-mask handles the rest).
        finite = torch.isfinite(attn).any(dim=-1, keepdim=True)
        attn = attn.softmax(dim=-1)
        attn = torch.where(finite, attn, torch.zeros_like(attn))
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.o(out)


class MaskableTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=12, head_dim=64):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MaskableSelfAttention(dim, num_heads, head_dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Cross-attention decoder for mode B
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
        B, Tq, _ = q.shape
        Tkv = kv.shape[1]
        Q = self.q(q).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(kv).view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(kv).view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if kv_mask is not None:
            block = ~kv_mask  # block where kv_mask is False
            attn = attn.masked_fill(block.unsqueeze(1).unsqueeze(2), float('-inf'))
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
        pe = self.pos_embed[0][unv_ids]  # (B, K, D)
        q = self.mask_token.expand(B, K, -1) + pe
        for blk in self.layers:
            q = blk(q, gaze_ctx, gaze_kv_mask)
        q = self.norm(q)
        return self.out_proj(q)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_unvisited(fix_ids, fix_mask, num_patches=49):
    """Returns (unv_ids: (B, P), unv_mask: (B, P)).
    For each row: valid unvisited IDs are placed first in raster order; rest
    are pad with mask=False."""
    B, T = fix_ids.shape
    fix_ids_clamped = fix_ids.clamp(0, num_patches - 1)
    one_hot = F.one_hot(fix_ids_clamped, num_classes=num_patches).bool()  # (B,T,P)
    visited_mat = (one_hot & fix_mask.unsqueeze(-1)).any(dim=1)  # (B, P)
    unvisited_mat = ~visited_mat
    P = num_patches
    grid = torch.arange(P, device=fix_ids.device).unsqueeze(0).expand(B, -1)
    sort_key = (~unvisited_mat).long() * P + grid
    sorted_idx = sort_key.argsort(dim=1)
    out_ids = grid.gather(1, sorted_idx)
    out_mask = unvisited_mat.gather(1, sorted_idx)
    return out_ids, out_mask


def patch_id_to_coord(patch_ids, grid_size=7):
    row = patch_ids // grid_size
    col = patch_ids % grid_size
    cx = (col.float() + 0.5) / grid_size
    cy = (row.float() + 0.5) / grid_size
    return torch.stack([cx, cy], dim=-1)


def pack_left(ids, coords, meta, mask):
    """Pack valid tokens to the left, preserving original relative order.
    ids:    (B, N) long
    coords: (B, N, 2)
    meta:   (B, N, M)
    mask:   (B, N) bool
    """
    B, N = ids.shape
    pos = torch.arange(N, device=ids.device).unsqueeze(0).expand(B, -1)
    sort_key = (~mask).long() * N + pos  # valid first; preserve original order
    order = sort_key.argsort(dim=1)  # (B, N)
    out_ids = ids.gather(1, order)
    out_coords = coords.gather(1, order.unsqueeze(-1).expand(-1, -1, coords.shape[-1]))
    out_meta = meta.gather(1, order.unsqueeze(-1).expand(-1, -1, meta.shape[-1]))
    out_mask = mask.gather(1, order)
    return out_ids, out_coords, out_meta, out_mask


# ---------------------------------------------------------------------------
# Main V4 model
# ---------------------------------------------------------------------------

class FixationARJEPA_V4(nn.Module):
    def __init__(
        self,
        mode='a',
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
        ar_weight=1.0,
    ):
        super().__init__()
        assert mode in ('a', 'b', 'c', 'd'), f"mode must be a|b|c|d, got {mode}"
        self.mode = mode
        self.completion_weight = completion_weight
        self.ar_weight = ar_weight

        # ---- backbone ----
        self.vit = MedCLIPSwinEncoder(medclip_ckpt)
        self.vit_dim = self.vit.embed_dim
        self.num_patches = self.vit.num_patches  # 49
        self.grid_size = self.vit.grid_size      # 7
        if not unfreeze_swin:
            for p in self.vit.parameters():
                p.requires_grad = False

        # ---- fixation embedder (max_seq covers gaze+unvisited for mode A) ----
        max_seq = max_fixations + self.num_patches
        self.fix_embedder = FixationEmbedder(
            vit_dim=self.vit_dim, embed_dim=embed_dim,
            max_fixations=max_seq, grid_size=self.grid_size,
        )

        # ---- causal context blocks (Maskable) ----
        self.context_blocks = nn.ModuleList([
            MaskableTransformerBlock(embed_dim, context_heads, head_dim)
            for _ in range(context_depth)
        ])
        self.context_norm = RMSNorm(embed_dim)

        # ---- predictor ----
        self.predictor = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # ---- EMA target for embedder/blocks/norm ----
        self.target_embedder = copy.deepcopy(self.fix_embedder)
        self.target_blocks = copy.deepcopy(self.context_blocks)
        self.target_norm = copy.deepcopy(self.context_norm)
        for p in (list(self.target_embedder.parameters()) +
                   list(self.target_blocks.parameters()) +
                   list(self.target_norm.parameters())):
            p.requires_grad = False

        # ---- mode-specific extras ----
        if mode == 'b':
            self.target_vit = copy.deepcopy(self.vit)
            for p in self.target_vit.parameters():
                p.requires_grad = False
            self.spatial_decoder = SpatialCompletionDecoder(
                embed_dim=embed_dim, num_heads=context_heads, head_dim=head_dim,
                depth=spatial_decoder_depth, num_patches=self.num_patches,
            )
        if mode == 'c':
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            self.mask_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.mask_pos_embed, std=0.02)

        # ---- eval (for SFT compatibility) ----
        self.eval_proj = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, self.vit_dim),
        )
        self.readout_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.readout_token, std=0.02)

        self.embed_dim = embed_dim
        self.ema_momentum = ema_momentum

    # ---------- shared utils ----------
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

    # ---------- dispatch ----------
    def forward(self, imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        if self.mode == 'a':
            return self._forward_a(imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask)
        if self.mode == 'b':
            return self._forward_b(imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask)
        if self.mode == 'c':
            return self._forward_c(imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask)
        if self.mode == 'd':
            return self._forward_d(imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask)

    # ---------- mode A ----------
    def _forward_a(self, imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        B = imgs.shape[0]
        patch_tokens = self._encode_image(imgs)
        gaze_ids = self._remap_patch_ids(fix_patch_ids, fix_coords)
        unv_ids, unv_mask = compute_unvisited(gaze_ids, fix_mask, self.num_patches)
        unv_coords = patch_id_to_coord(unv_ids, self.grid_size)
        unv_meta = torch.zeros(B, self.num_patches, fix_meta.shape[-1],
                               device=imgs.device, dtype=fix_meta.dtype)

        all_ids = torch.cat([gaze_ids, unv_ids], dim=1)
        all_coords = torch.cat([fix_coords, unv_coords], dim=1)
        all_meta = torch.cat([fix_meta, unv_meta], dim=1)
        all_mask = torch.cat([fix_mask, unv_mask], dim=1)
        all_ids, all_coords, all_meta, all_mask = pack_left(
            all_ids, all_coords, all_meta, all_mask)

        ctx_embeds = self.fix_embedder(
            patch_tokens, all_ids, all_coords, all_meta, all_mask)
        x = ctx_embeds
        for blk in self.context_blocks:
            x = blk(x, attn_mask=None)  # default causal
        x = self.context_norm(x)
        preds = self.predictor(x[:, :-1, :])

        with torch.no_grad():
            tgt = self.target_embedder(
                patch_tokens.detach(), all_ids, all_coords, all_meta, all_mask)
            y = tgt
            for blk in self.target_blocks:
                y = blk(y, attn_mask=None)
            y = self.target_norm(y)
            targets = y[:, 1:, :]
            targets = F.layer_norm(targets, [targets.shape[-1]])

        valid = all_mask[:, 1:].unsqueeze(-1).float()
        return {'ar_preds': preds * valid, 'ar_targets': targets * valid}

    # ---------- mode B ----------
    def _forward_b(self, imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        patch_tokens = self._encode_image(imgs)
        gaze_ids = self._remap_patch_ids(fix_patch_ids, fix_coords)

        # AR head (V3-style)
        ctx_embeds = self.fix_embedder(
            patch_tokens, gaze_ids, fix_coords, fix_meta, fix_mask)
        x = ctx_embeds
        for blk in self.context_blocks:
            x = blk(x, attn_mask=None)
        x = self.context_norm(x)
        preds = self.predictor(x[:, :-1, :])

        with torch.no_grad():
            tgt = self.target_embedder(
                patch_tokens.detach(), gaze_ids, fix_coords, fix_meta, fix_mask)
            y = tgt
            for blk in self.target_blocks:
                y = blk(y, attn_mask=None)
            y = self.target_norm(y)
            targets = y[:, 1:, :]
            targets = F.layer_norm(targets, [targets.shape[-1]])

        valid = fix_mask[:, 1:].unsqueeze(-1).float()
        ar_preds = preds * valid
        ar_targets = targets * valid

        # Spatial completion head
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

    # ---------- mode C ----------
    def _forward_c(self, imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        B, T_gaze = fix_patch_ids.shape
        P = self.num_patches
        patch_tokens = self._encode_image(imgs)
        gaze_ids = self._remap_patch_ids(fix_patch_ids, fix_coords)

        gaze_tokens = self.fix_embedder(
            patch_tokens, gaze_ids, fix_coords, fix_meta, fix_mask)

        unv_ids, unv_mask = compute_unvisited(gaze_ids, fix_mask, P)
        mask_pe = self.mask_pos_embed[0][unv_ids]
        mask_tokens = (self.mask_token.expand(B, P, -1) + mask_pe)
        mask_tokens = mask_tokens * unv_mask.unsqueeze(-1).float()

        x = torch.cat([gaze_tokens, mask_tokens], dim=1)
        N = T_gaze + P

        # Build attention mask (True = block)
        causal_g = torch.triu(torch.ones(T_gaze, T_gaze, device=imgs.device, dtype=torch.bool),
                              diagonal=1)
        block_g_to_m = torch.ones(T_gaze, P, device=imgs.device, dtype=torch.bool)
        block_m_to_g = torch.zeros(P, T_gaze, device=imgs.device, dtype=torch.bool)
        block_m_to_m = ~torch.eye(P, device=imgs.device, dtype=torch.bool)
        attn_mask_2d = torch.cat([
            torch.cat([causal_g, block_g_to_m], dim=1),
            torch.cat([block_m_to_g, block_m_to_m], dim=1),
        ], dim=0)  # (N, N)
        col_valid = torch.cat([fix_mask, unv_mask], dim=1)  # (B, N)
        attn_mask = attn_mask_2d.unsqueeze(0).expand(B, -1, -1).clone()
        attn_mask = attn_mask | (~col_valid).unsqueeze(1)

        for blk in self.context_blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.context_norm(x)

        gaze_x, mask_x = x[:, :T_gaze, :], x[:, T_gaze:, :]
        ar_preds = self.predictor(gaze_x[:, :-1, :])
        spat_preds = self.predictor(mask_x)

        with torch.no_grad():
            tgt = self.target_embedder(
                patch_tokens.detach(), gaze_ids, fix_coords, fix_meta, fix_mask)
            y = tgt
            for blk in self.target_blocks:
                y = blk(y, attn_mask=None)
            y = self.target_norm(y)
            ar_targets = y[:, 1:, :]
            ar_targets = F.layer_norm(ar_targets, [ar_targets.shape[-1]])

        valid_ar = fix_mask[:, 1:].unsqueeze(-1).float()
        ar_preds = ar_preds * valid_ar
        ar_targets = ar_targets * valid_ar

        with torch.no_grad():
            idx = unv_ids.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
            spat_targets_raw = torch.gather(patch_tokens.detach(), 1, idx)
            spat_targets = F.layer_norm(spat_targets_raw, [spat_targets_raw.shape[-1]])

        valid_unv = unv_mask.unsqueeze(-1).float()
        return {'ar_preds': ar_preds, 'ar_targets': ar_targets,
                'spat_preds': spat_preds * valid_unv,
                'spat_targets': spat_targets * valid_unv,
                'spat_mask': unv_mask}

    # ---------- mode D ----------
    def _forward_d(self, imgs, fix_patch_ids, fix_coords, fix_meta, fix_mask):
        """[unvisited prefix (bidirectional), gaze suffix (causal)]; AR loss on gaze.

        Unvisited tokens carry MedCLIP-Swin patch features at their grid
        positions (the model has 'seen' the whole image), grid-cell-center
        coords, and a flag in fix_meta marking them as 'context, not gaze'.
        """
        B, T_gaze = fix_patch_ids.shape
        P = self.num_patches
        patch_tokens = self._encode_image(imgs)
        gaze_ids = self._remap_patch_ids(fix_patch_ids, fix_coords)

        # Unvisited prefix: synthetic fixation tuples for fix_embedder
        unv_ids, unv_mask = compute_unvisited(gaze_ids, fix_mask, P)
        unv_coords = patch_id_to_coord(unv_ids, self.grid_size)
        unv_meta = torch.zeros(B, P, fix_meta.shape[-1],
                               device=imgs.device, dtype=fix_meta.dtype)

        all_ids = torch.cat([unv_ids, gaze_ids], dim=1)
        all_coords = torch.cat([unv_coords, fix_coords], dim=1)
        all_meta = torch.cat([unv_meta, fix_meta], dim=1)
        all_mask = torch.cat([unv_mask, fix_mask], dim=1)
        # Pack so prefix-real, gaze-real, pad — preserve relative order
        all_ids, all_coords, all_meta, all_mask = pack_left(
            all_ids, all_coords, all_meta, all_mask)

        ctx = self.fix_embedder(
            patch_tokens, all_ids, all_coords, all_meta, all_mask)

        # Build prefix-LM attention mask per sample.
        # Per-sample: prefix length K_b = unv_mask[b].sum(); gaze length T_b = fix_mask[b].sum().
        # Within prefix: bidirectional. Within gaze: causal. Gaze attends to all prefix.
        # Prefix attends to gaze: blocked.
        N = all_ids.shape[1]
        # Number of valid prefix tokens per sample
        K_b = unv_mask.sum(dim=1)  # (B,)
        # Position index in [0, N) along the packed sequence
        col = torch.arange(N, device=imgs.device).unsqueeze(0).expand(B, -1)  # (B, N)
        row = col  # same indices for query rows
        # is_gaze[b, i] = True iff position i is a gaze token (i.e., i >= K_b[b])
        is_gaze_row = row >= K_b.unsqueeze(1)
        is_gaze_col = col >= K_b.unsqueeze(1)
        # Block matrix:
        #   prefix→prefix : allow (False)
        #   prefix→gaze   : block (True)
        #   gaze→prefix   : allow
        #   gaze→gaze     : causal
        block = torch.zeros(B, N, N, device=imgs.device, dtype=torch.bool)
        # prefix rows attending to gaze cols
        block |= (~is_gaze_row.unsqueeze(2)) & is_gaze_col.unsqueeze(1)
        # gaze rows attending to gaze cols where col_idx_within_gaze > row_idx_within_gaze
        causal_full = torch.triu(torch.ones(N, N, device=imgs.device, dtype=torch.bool),
                                 diagonal=1).unsqueeze(0).expand(B, -1, -1)
        gaze_pair = is_gaze_row.unsqueeze(2) & is_gaze_col.unsqueeze(1)
        block |= gaze_pair & causal_full
        # Mask out padding columns
        block |= (~all_mask).unsqueeze(1)

        x = ctx
        for blk in self.context_blocks:
            x = blk(x, attn_mask=block)
        x = self.context_norm(x)

        # Predictor on every position; loss applied only to "predict next gaze" pairs:
        # at position i, we predict position i+1 — but we want loss only when
        # both i and i+1 are gaze tokens (so we're predicting a gaze step).
        preds = self.predictor(x[:, :-1, :])

        with torch.no_grad():
            tgt = self.target_embedder(
                patch_tokens.detach(), all_ids, all_coords, all_meta, all_mask)
            y = tgt
            for blk in self.target_blocks:
                y = blk(y, attn_mask=None)  # target uses default causal
            y = self.target_norm(y)
            targets = y[:, 1:, :]
            targets = F.layer_norm(targets, [targets.shape[-1]])

        # Loss mask: valid AND both ends are gaze
        gaze_at_t = is_gaze_row & all_mask
        loss_mask = gaze_at_t[:, :-1] & gaze_at_t[:, 1:]  # (B, N-1)
        valid = loss_mask.unsqueeze(-1).float()
        return {'ar_preds': preds * valid, 'ar_targets': targets * valid}

    # ---------- loss ----------
    def compute_loss(self, out):
        ar_preds = F.layer_norm(out['ar_preds'], [out['ar_preds'].shape[-1]])
        l_ar = F.smooth_l1_loss(ar_preds, out['ar_targets'])
        result = {'l_ar': l_ar.detach()}
        if self.mode in ('b', 'c'):
            sp = F.layer_norm(out['spat_preds'], [out['spat_preds'].shape[-1]])
            l_sp = F.smooth_l1_loss(sp, out['spat_targets'])
            result['l_sp'] = l_sp.detach()
            result['loss'] = self.ar_weight * l_ar + self.completion_weight * l_sp
        else:
            result['loss'] = self.ar_weight * l_ar
        return result

    # ---------- EMA ----------
    @torch.no_grad()
    def update_target(self, momentum=None):
        m = momentum if momentum is not None else self.ema_momentum
        pairs = [
            (self.fix_embedder.parameters(), self.target_embedder.parameters()),
            (self.context_blocks.parameters(), self.target_blocks.parameters()),
            (self.context_norm.parameters(), self.target_norm.parameters()),
        ]
        for src, tgt in pairs:
            for p_s, p_t in zip(src, tgt):
                p_t.data.mul_(m).add_((1 - m) * p_s.data)
        if self.mode == 'b':
            for p_s, p_t in zip(self.vit.parameters(), self.target_vit.parameters()):
                p_t.data.mul_(m).add_((1 - m) * p_s.data)

    # ---------- features for SFT (mirrors V3 contract) ----------
    def get_features(self, imgs):
        patch_tokens = self._encode_image(imgs)
        B = patch_tokens.shape[0]
        G = self.grid_size
        T_scan = self.num_patches
        fix_ids = torch.arange(T_scan, device=imgs.device).unsqueeze(0).expand(B, -1)
        fix_coords = torch.zeros(B, T_scan, 2, device=imgs.device)
        for i in range(T_scan):
            fix_coords[:, i, 0] = (i % G) / G
            fix_coords[:, i, 1] = (i // G) / G
        fix_meta = torch.zeros(B, T_scan, 3, device=imgs.device)
        fix_meta[:, :, 2] = torch.linspace(0, 1, T_scan, device=imgs.device)
        fix_mask = torch.ones(B, T_scan, dtype=torch.bool, device=imgs.device)
        ctx = self.fix_embedder(patch_tokens, fix_ids, fix_coords, fix_meta, fix_mask)
        readout = self.readout_token.expand(B, -1, -1)
        x = torch.cat([readout, ctx], dim=1)
        for blk in self.context_blocks:
            x = blk(x, attn_mask=None)
        x = self.context_norm(x)
        feat = self.eval_proj(x[:, 0, :])
        vit_feat = patch_tokens.mean(dim=1)
        return torch.cat([feat, vit_feat], dim=-1)


def build_fixation_ar_jepa_v4(cfg, mode):
    m = cfg["model"]
    return FixationARJEPA_V4(
        mode=mode,
        embed_dim=m.get("embed_dim", 768),
        context_depth=m.get("context_depth", 8),
        context_heads=m.get("context_heads", 12),
        head_dim=m.get("head_dim", 64),
        max_fixations=m.get("max_fixations", 96),
        ema_momentum=m.get("ema_momentum", 0.998),
        medclip_ckpt=m.get("medclip_ckpt",
            "/tmp/medclip_weights/medclip-vit-pretrained/pytorch_model.bin"),
        unfreeze_swin=m.get("unfreeze_swin", True),
        spatial_decoder_depth=m.get("spatial_decoder_depth", 2),
        completion_weight=m.get("completion_weight", 1.0),
        ar_weight=m.get("ar_weight", 1.0),
    )

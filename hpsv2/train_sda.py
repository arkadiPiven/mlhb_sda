#!/usr/bin/env python
import sys
import json
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np

from losses import PairwiseHingeRankingLoss

import matplotlib.pyplot as plt
import io

from transformers import get_cosine_schedule_with_warmup

class PermEqui2_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui2_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        # print(x.size())
        xm, _ = x.max(1, keepdim=True)
        # print(xm.size())
        xm = self.Lambda(xm)
        # print(xm.size())
        x = self.Gamma(x)
        x = x - xm
        return x
    
class DeepSetsTanh(nn.Module):
    def __init__(self, d_dim, x_dim=3, keep_prob=0.5, invariant=True):
        super(DeepSetsTanh, self).__init__()
        self.d_dim = d_dim
        self.x_dim = x_dim
        self.invariant = invariant

        self.phi = nn.Sequential(
            PermEqui2_max(self.x_dim, self.d_dim),
            nn.Tanh(),
            PermEqui2_max(self.d_dim, self.d_dim),
            nn.Tanh(),
            PermEqui2_max(self.d_dim, self.d_dim),
            nn.Tanh(),
        )

        self.ro = nn.Sequential(
            nn.Dropout(p=keep_prob),
            nn.Linear(self.d_dim, self.d_dim),
            nn.Tanh(),
            nn.Dropout(p=keep_prob),
            nn.Linear(self.d_dim, self.d_dim),
        )
        # print(self)

    def forward(self, x):
        phi_output = self.phi(x)
        ro_output = self.ro(phi_output)
        if self.invariant:
            return ro_output.sum(dim=1, keepdim=True)
        else:
            return ro_output

class KinkedTanh(nn.Module):
    r"""
    μ(z; c) = tanh(z) · (c  if z < 0
                         1  otherwise),   with  c ≥ 1  and learnable.
    """
    def __init__(self, c: float = 2.0, min_c: float = 1.0) -> None:
        """
        Args
        ----
        init_c : initial value for the slope on z<0  (must be ≥ min_c)
        min_c  : hard lower bound that c is not allowed to cross
        """
        super().__init__()
        
        init_c = c
        if init_c < min_c:
            raise ValueError(f"init_c ({init_c}) must be ≥ min_c ({min_c}).")

        # ── unconstrained parameter ───────────────────────────────────────
        # We store log( c - min_c ) so that   c = min_c + exp(param) ≥ min_c.
        self._log_c_offset = nn.Parameter(
            torch.log(torch.tensor(init_c - min_c, dtype=torch.float32))
        )
        self.min_c = min_c

    @property
    def c(self) -> torch.Tensor:
        """Current value of the slope parameter (always ≥ min_c)."""
        return self.min_c + torch.exp(self._log_c_offset)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z<0 → multiply by c ;  z≥0 → multiply by 1
        slope = torch.where(z < 0, self.c, torch.ones_like(z))
        return torch.tanh(z) * slope


class FinalSDA(nn.Module):
    def __init__(self, d_in, l):
        super(FinalSDA, self).__init__()
        self.d_in = d_in
        self.l = l

        self.F = nn.Sequential(
            nn.Linear(2*d_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, l)
        )

        self.w_set_nn = SetTransformer(
            dim_input=l,                
            num_outputs=1,              
            dim_output=l,  
            num_inds=16,   
            dim_hidden=512,
            num_heads=8,
            ln=True
        )
        self.r = SetTransformer(
            dim_input=l,            
            num_outputs=1,          
            dim_output=l,  
            num_inds=16,   
            dim_hidden=512,
            num_heads=8,
            ln=True
        )

        self.kinked = KinkedTanh(c=2.0)
    
    def forward(self, image_features, text_features, num_images, scale):
        """
        Args:
            image_features: Tensor of shape (total_images, clip_dim)
            text_features: Tensor of shape (B, clip_dim)
            num_images: Tensor or list with number of images per example (length B)
            scale: CLIP logit scale (not used in this snippet, but available if needed)
        Returns:
            List of score tensors (each of shape (n_i,)) for each example in the batch.
        """
        B = text_features.size(0)

        image_features = image_features.view(B, -1, 1024)

        text_features_expanded = text_features.unsqueeze(1).expand(-1, image_features.size(1), -1)
        features = torch.cat((image_features, text_features_expanded), dim=2)

        F_x = self.F(features.view(-1, 2048))

        F_x = F_x.view(B, -1, self.l)
        

        w_l = self.w_set_nn(F_x)

        phi_l = self.kinked(F_x.reshape(B, -1, self.l) - self.r(F_x.reshape(B, -1, self.l)))

        scores = torch.sum(w_l * phi_l, dim=2)

        return scores

class SDW(nn.Module):
    def __init__(self, d_in, l):
        super(SDW, self).__init__()
        self.d_in = d_in
        self.l = l

        self.F = nn.Sequential(
            nn.Linear(2*d_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, l)
        )

        self.w_set_nn = DeepSetsTanh(l, x_dim=l)

        self.w_transformer = SetTransformer(
            dim_input=l,                # input dimension matches set representation
            num_outputs=1,                         # produce a single output per example
            dim_output=l,  # output dimension matches desired flattened weight vector
            num_inds=16,                           # number of inducing points (adjust as needed)
            dim_hidden=512,
            num_heads=8,
            ln=True
        )
    

    def forward(self, image_features, text_features, num_images, scale):
        """
        Args:
            image_features: Tensor of shape (total_images, clip_dim)
            text_features: Tensor of shape (B, clip_dim)
            num_images: Tensor or list with number of images per example (length B)
            scale: CLIP logit scale (not used in this snippet, but available if needed)
        Returns:
            List of score tensors (each of shape (n_i,)) for each example in the batch.
        """
        B = text_features.size(0)

        image_features = image_features.view(B, -1, 1024)

        text_features_expanded = text_features.unsqueeze(1).expand(-1, image_features.size(1), -1)
        features = torch.cat((image_features, text_features_expanded), dim=2)

        F_x = self.F(features)
        w_l = self.w_transformer(F_x)

        # print(F_x.size())
        # print(w_l.size())

        scores = torch.sum(F_x * w_l, dim=2)
        # print(scores.size())
        return scores

class SDE(nn.Module):
    def __init__(self, d_in, l):
        super(SDE, self).__init__()
        self.d_in = d_in
        self.l = l

        self.F = nn.Sequential(
            nn.Linear(2*d_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, l)
        )


        self.w_set_nn = DeepSetsTanh(l, x_dim=l)
        self.phi = DeepSetsTanh(l, x_dim=l)
        self.w_transformer = SetTransformer(
            dim_input=l,                # input dimension matches set representation
            num_outputs=1,                         # produce a single output per example
            dim_output=l,  # output dimension matches desired flattened weight vector
            num_inds=16,                           # number of inducing points (adjust as needed)
            dim_hidden=512,
            num_heads=8,
            ln=True
        )

        self.phi_transformer = SetTransformer(
            dim_input=l,                # input dimension matches set representation
            num_outputs=1,                         # produce a single output per example
            dim_output=l,  # output dimension matches desired flattened weight vector
            num_inds=16,                           # number of inducing points (adjust as needed)
            dim_hidden=512,
            num_heads=8,
            ln=True
        )

    def forward(self, image_features, text_features, num_images, scale):
        """
        Args:
            image_features: Tensor of shape (total_images, clip_dim)
            text_features: Tensor of shape (B, clip_dim)
            num_images: Tensor or list with number of images per example (length B)
            scale: CLIP logit scale (not used in this snippet, but available if needed)
        Returns:
            List of score tensors (each of shape (n_i,)) for each example in the batch.
        """
        B = text_features.size(0)

        image_features = image_features.view(B, -1, 1024)

        text_features_expanded = text_features.unsqueeze(1).expand(-1, image_features.size(1), -1)
        features = torch.cat((image_features, text_features_expanded), dim=2)

        F_x = self.F(features)
        w_l = self.w_transformer(F_x)
        phi_l = self.phi_transformer(F_x)

        scores = torch.sum(w_l * phi_l, dim=2)
        return scores


def set_openclip_trainability(model,
                              img_blocks_to_train: int = 20,
                              txt_blocks_to_train: int = 11):
    """
    Freeze all parameters in an OpenCLIP model except
    - the last `img_blocks_to_train` blocks of the image transformer
    - the last `txt_blocks_to_train` blocks of the text transformer
    - the projection heads and logit scale

    Parameters
    ----------
    model : open_clip.CLIP
        The loaded OpenCLIP model (e.g. ViT‑H/14).
    img_blocks_to_train : int
        How many *final* image‑transformer blocks to unfreeze.
    txt_blocks_to_train : int
        How many *final* text‑transformer blocks to unfreeze.
    """

    # ---------- 0. Freeze everything ----------
    for p in model.parameters():
        p.requires_grad = False

    # ---------- 1. Image encoder ----------
    vt = model.visual.transformer          # Vision Transformer
    img_blocks = getattr(vt, "resblocks", getattr(vt, "blocks", None))
    if img_blocks is None:
        raise RuntimeError("Cannot locate image transformer blocks")

    for blk in img_blocks[-img_blocks_to_train:]:
        for p in blk.parameters():
            p.requires_grad = True

    # final image projection (d_model → 1024)
    if hasattr(model.visual, "proj") and model.visual.proj is not None:
        model.visual.proj.requires_grad = True

    # ---------- 2. Text encoder ----------
    tt = getattr(model, "transformer", getattr(model, "text", None))
    if tt is None:
        raise RuntimeError("Cannot locate text transformer")

    txt_blocks = getattr(tt, "resblocks", getattr(tt, "blocks", None))
    if txt_blocks is None:
        raise RuntimeError("Cannot locate text transformer blocks")

    for blk in txt_blocks[-txt_blocks_to_train:]:
        for p in blk.parameters():
            p.requires_grad = True

    # final text projection (d_model → 1024)
    if hasattr(model, "text_projection") and model.text_projection is not None:
        model.text_projection.requires_grad = True

    # ---------- 3. Logit‑scale (always useful to fine‑tune) ----------
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True

    # ---------- 4. (Optional) sanity check ----------
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable/1e6:.2f} M / {total/1e6:.2f} M "
          f"({trainable/total:.1%})")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# If you need to import local open_clip from a "src" folder, do something like:
# sys.path.insert(0, "src")
from src.open_clip import create_model_and_transforms, get_tokenizer
from src.training.data import collate_rank, RankingDataset
from torch.utils.data import DataLoader
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import wandb

# Initialize wandb
wandb.init(
    project="sda",
    name="fix_sda",
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
def top1_accuracy(scores, ranks):
    """
    Check if the item with rank=0 also has the highest predicted score.
    """
    best_idx = (ranks == 0).nonzero(as_tuple=True)
    if len(best_idx[0]) < 1:
        return 0.0
    best_idx = best_idx[0][0].item()
    pred_idx = torch.argmax(scores).item()
    return 1.0 if pred_idx == best_idx else 0.0


def spearman_rho_torch(scores, ranks):
    n = scores.shape[0]
    if n < 2:
        return 0.0
    predicted_order = torch.argsort(scores, descending=True)
    predicted_rank = torch.zeros_like(ranks, dtype=torch.float)
    for rankpos, item_idx in enumerate(predicted_order):
        predicted_rank[item_idx] = rankpos
    gt_rank = ranks.float()
    d = predicted_rank - gt_rank
    d2 = (d ** 2).sum()
    denom = n * (n ** 2 - 1)
    rho = 1.0 - (6.0 * d2) / denom
    return rho.item()


def compute_ndcg(scores: torch.Tensor, ranks: torch.Tensor):
    max_rank = ranks.max().item()
    relevances = max_rank - ranks  # Higher relevance for lower rank number
    relevances = relevances.detach().cpu().numpy().reshape(1, -1)
    predicted = scores.detach().cpu().numpy().reshape(1, -1)
    ndcg_val = ndcg_score(relevances, predicted)
    return ndcg_val


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        """
        A simple DeepSet implementation.
        It encodes a set of input elements and then aggregates them via a mean,
        followed by a decoder that maps to a fixed-size output.
        """
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )

    def forward(self, X):
        # X is expected to be of shape (T, dim_input) where T is the number of elements.
        # Encode each element.
        X_enc = self.enc(X)  # shape: (T, dim_hidden)
        # Aggregate via mean over the set dimension.
        X_pool = X_enc.mean(dim=0, keepdim=True)  # shape: (1, dim_hidden)
        # Decode to get fixed-length representation.
        out = self.dec(X_pool)  # shape: (1, num_outputs*dim_output)
        out = out.view(-1, self.num_outputs, self.dim_output)  # shape: (1, num_outputs, dim_output)
        return out  # e.g., (1, 1, dim_output)


##############################################################################
# KinkedTanh as before.
##############################################################################

class ArkadiSDA(nn.Module):
    def __init__(self, clip_dim=1024, hidden_dim=512, embedding_dim=1024, l=24):
        super().__init__()
        """
        Implements the set-dependent scoring function using DeepSet for set processing.
        Args:
            clip_dim: Dimension of CLIP embeddings
            hidden_dim: Dimension of hidden layers
            embedding_dim: Dimension of the embedding space for inner products
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # w_layers = [
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # ] * l

        # self.w = nn.Sequential(
        #     *w_layers
        # )

        # r_layers = [
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # ] * l

        # self.r = nn.Sequential(
        #     *r_layers
        # )

        # Base function network: maps image features to embedding space
        self.F = nn.Sequential(
            nn.Linear(2 * clip_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # self.F = SetTransformer(
        #     dim_input=2 * clip_dim,
        #     dim_hidden=1024,
        #     dim_output=clip_dim,
        #     num_outputs=4,
        #     ln=True
        # )

        self.w = SetTransformer(
            dim_input=clip_dim,
            dim_hidden=1024,
            dim_output=clip_dim,
            num_outputs=1,
            ln=True
        )

        self.r = SetTransformer(
            dim_input=clip_dim,
            dim_hidden=1024,
            dim_output=clip_dim,
            num_outputs=1,
            ln=True
        )

        self.kink = KinkedTanh(c=2.0)

        self.full_transformer = SetTransformer(
            dim_input=2*clip_dim,
            dim_hidden=64,
            num_inds=16,
            num_heads=8,
            dim_output=1,
            num_outputs=4,
            ln=True
        )
    
    def forward(self, image_features, text_features, num_images, scale):
        B = text_features.size(0)
        image_features = image_features.reshape(B, -1, 1024)
        text_features = text_features.reshape(B, 1024)

        

        # Concatenate each image feature with the text feature
        features = torch.cat([image_features, text_features.unsqueeze(1).expand(-1, image_features.size(1), -1)], dim=-1)

        # w = self.w(self.F(features))
        # w = w.squeeze(1)
        # # print(f" W: {w.size()}")
        # # print(f" W: {w.size()}")
        # r = self.r(self.F(features))
        # r = r.squeeze(1)
        # # print(f" R: {r.size()}")
        # F_x = self.F(features)
        # # print(f" F_X: {F_x.size()}")
        # phi = self.kink(F_x - r[:, None, :])
        # # print(f" PHI: {phi.size()}")
        # scalar_products = torch.einsum('bkd,b d->bk', phi, w)

        # # print(f" SCALAR PRODUCTS: {scalar_products.size()}")
        # return scalar_products
        # # print(w.size())
        # # print(r.size())
        # # print(F_x.size())
        
        
        return self.full_transformer(features).squeeze(-1)



class SetDependentScorer(nn.Module):
    def __init__(self, clip_dim=1024, hidden_dim=512, num_functions=24, embedding_dim=1024):
        """
        Implements the set-dependent scoring function using SetTransformer for set processing.
        Args:
            clip_dim: Dimension of CLIP embeddings
            hidden_dim: Dimension of hidden layers
            num_functions: Number of scoring functions to combine
            embedding_dim: Dimension of the embedding space for inner products
        """
        super().__init__()
        self.num_functions = num_functions
        self.embedding_dim = embedding_dim

        # Base function network: maps image features to embedding space
        self.F = nn.Sequential(
            nn.Linear(2 * clip_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_functions * embedding_dim)
        )

        # Replace DeepSet with SetTransformer for set-level representation
        # num_outputs=1 because we want a single set representation
        self.set_transformer = SetTransformer(
            dim_input=2*clip_dim,
            num_outputs=1,  # Single set representation
            dim_output=2*clip_dim,  # Match CLIP dimension
            num_inds=16,  # Number of inducing points
            dim_hidden=hidden_dim,
            num_heads=48,
            ln=True  # Use layer normalization
        )

        # Reference network: produces reference vectors r(s)
        # self.r_net = nn.Sequential(
        #     nn.Linear(2*clip_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_functions * embedding_dim)
        # )

        self.r_net = SetTransformer(
            dim_input=2 * clip_dim,                     # Input dimension matches the set representation
            num_outputs=1,                              # We want one output per example
            dim_output=num_functions * embedding_dim,   # Flattened output size
            num_inds=16,                                # Number of inducing points (adjust as needed)
            dim_hidden=hidden_dim,
            num_heads=8,
            ln=True
        )

        # Weight network: produces weight vectors w(s)
        # self.w_net = nn.Sequential(
        #     nn.Linear(2*clip_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_functions * embedding_dim),
        #     nn.Softplus()  # Ensure positive weights
        # )

        self.w_net = SetTransformer(
            dim_input=2 * clip_dim,                # input dimension matches set representation
            num_outputs=1,                         # produce a single output per example
            dim_output=num_functions * embedding_dim,  # output dimension matches desired flattened weight vector
            num_inds=16,                           # number of inducing points (adjust as needed)
            dim_hidden=hidden_dim,
            num_heads=8,
            ln=True
        )

        self.kink = KinkedTanh(c=2.0)

    def compute_inner_product(self, a, b):
        """
        Compute inner product between batched vectors.
        Args:
            a: Tensor of shape (..., num_functions, embedding_dim)
            b: Tensor of shape (..., num_functions, embedding_dim)
        Returns:
            Tensor of shape (..., num_functions)
        """
        return torch.sum(a * b, dim=-1)  # Sum over embedding dimension

    def forward(self, image_features, text_features, num_images, scale):
        """
        Args:
            image_features: Tensor of shape (total_images, clip_dim)
            text_features: Tensor of shape (B, clip_dim)
            num_images: Tensor or list with number of images per example (length B)
            scale: CLIP logit scale (not used in this snippet, but available if needed)
        Returns:
            List of score tensors (each of shape (n_i,)) for each example in the batch.
        """
        # Ensure num_images is a list
        if not isinstance(num_images, list):
            num_images = num_images.tolist()
        
        # 1. Split image_features into a list by example.
        image_batches = torch.split(image_features, num_images)
        B = len(image_batches)

        # 2. For each example, expand its text feature to match the number of images and concatenate.
        concatenated_list = []
        for i in range(B):
            n = num_images[i]
            # Expand the text feature for example i: (clip_dim,) -> (n, clip_dim)
            text_expanded = text_features[i].unsqueeze(0).expand(n, -1)
            # Concatenate along the feature dimension → (n, 2*clip_dim)
            concatenated = torch.cat([image_batches[i], text_expanded], dim=1)
            concatenated_list.append(concatenated)

        # 3. Pad the list into a tensor of shape (B, T_max, 2*clip_dim) and build a mask.
        padded_features = torch.nn.utils.rnn.pad_sequence(concatenated_list, batch_first=True)
        B, T_max, _ = padded_features.shape
        lengths = torch.tensor(num_images, device=padded_features.device)
        # mask[i, j] = True if j is a valid index for example i.
        mask = torch.arange(T_max, device=padded_features.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # 4. Compute F(x) for all images in one go.
        # Flatten the padded tensor: (B*T_max, 2*clip_dim)
        flat_features = padded_features.view(B * T_max, -1)
        F_x = self.F(flat_features)  # (B*T_max, num_functions * embedding_dim)
        F_x = F_x.view(B, T_max, self.num_functions, self.embedding_dim)  # (B, T_max, num_functions, embedding_dim)

        # 5. Compute a set-level representation per example using the set transformer.
        #    (Assumes self.set_transformer can accept a mask. Otherwise, consider using masked pooling.)
        try:
            # Expected output shape: (B, 1, clip_dim)
            set_repr = self.set_transformer(padded_features, mask=mask)
        except Exception:
            # Fallback: simple mean pooling over valid positions.
            set_repr = (padded_features * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1).float()
            set_repr = set_repr.unsqueeze(1)
        set_repr = set_repr.squeeze(1)  # (B, clip_dim)

        # 6. Compute w(s) and r(s) for each example.
        w_s = self.w_net(set_repr.unsqueeze(1))  # Now shape: (B, 1, num_functions*embedding_dim)
        w_s = w_s.squeeze(1)                     # Shape: (B, num_functions*embedding_dim)
        w_s = w_s.view(-1, self.num_functions, self.embedding_dim)  # (B, num_functions, embedding_dim)

        # Assume set_repr has shape (B, 2*clip_dim)
        r_s = self.r_net(set_repr.unsqueeze(1))  # Now shape: (B, 1, num_functions*embedding_dim)
        r_s = r_s.squeeze(1)                     # Shape: (B, num_functions*embedding_dim)
        r_s = r_s.view(-1, self.num_functions, self.embedding_dim)  # Reshape to (B, num_functions, embedding_dim)


        # w_s = self.w_net(set_repr)  # (B, num_functions * embedding_dim)
        # r_s = self.r_net(set_repr)  # (B, num_functions * embedding_dim)
        # # w_s = w_s.view(B, self.num_functions, self.embedding_dim)  # (B, num_functions, embedding_dim)
        # r_s = r_s.view(B, self.num_functions, self.embedding_dim)  # (B, num_functions, embedding_dim)



        # 7. SDW mechanism: compute inner product 〈w(s), F(x)〉.
        #    Expand w_s to (B, T_max, num_functions, embedding_dim) to match F_x.
        w_s_exp = w_s.unsqueeze(1).expand(B, T_max, self.num_functions, self.embedding_dim)
        scores_w = self.compute_inner_product(w_s_exp, F_x)  # (B, T_max, num_functions)

        # 8. SDE mechanism: compute μ(F(x) − r(s)).
        r_s_exp = r_s.unsqueeze(1).expand(B, T_max, self.num_functions, self.embedding_dim)
        diff = F_x - r_s_exp
        # Apply the non-linearity (kink) then mean over the embedding dimension.
        scores_e = self.kink(diff).mean(dim=3)  # (B, T_max, num_functions)

        # 9. Combine the two scores and collapse the function dimension.
        final_scores = scores_w + scores_e         # (B, T_max, num_functions)
        final_scores = final_scores.sum(dim=2)        # (B, T_max)

        # (Optional) You can apply scale here if needed, e.g.:
        # final_scores = final_scores * scale

        # 10. Remove padded positions: split final_scores into a list per example.
        batch_scores = [final_scores[i, :num].contiguous() for i, num in enumerate(num_images)]
        return batch_scores


    def get_intermediate_outputs(self, image_features, text_features, num_images, scale):
        """
        Helper method to get intermediate outputs for analysis.
        """
        image_batches = torch.split(image_features, num_images.tolist())
        outputs = []

        for img_feats, txt_feat in zip(image_batches, text_features):
            # Add batch dimension for SetTransformer
            img_feats_batch = img_feats.unsqueeze(0)
            set_repr = self.set_transformer(img_feats_batch)
            set_repr = set_repr.squeeze(0).squeeze(0)

            w_s = self.w_net(set_repr)
            r_s = self.r_net(set_repr)

            F_x = self.F(img_feats)
            F_x = F_x.view(-1, self.num_functions, self.embedding_dim)

            # Reshape for computations
            w_s = w_s.view(self.num_functions, self.embedding_dim)
            r_s = r_s.view(self.num_functions, self.embedding_dim)

            # Compute scores
            diff = F_x - r_s.unsqueeze(0)
            kinked_diff = self.kink(diff)

            scores_w = self.compute_inner_product(
                w_s.unsqueeze(0).expand(F_x.size(0), -1, -1),
                F_x
            )
            scores_e = kinked_diff.mean(dim=2)
            final_scores = scores_w + scores_e
            final_scores = final_scores.sum(dim=1)

            outputs.append({
                'set_representation': set_repr.detach(),
                'weights': w_s.detach(),
                'references': r_s.detach(),
                'base_features': F_x.detach(),
                'kinked_diff': kinked_diff.detach(),
                'final_scores': final_scores.detach()
            })

        return outputs

##############################################################################
# LISTMLE LOSS FUNCTION (Listwise Loss)
##############################################################################

# def listmle_loss(scores, true_ranks):
#     """
#     Computes the ListMLE loss for a single example.

#     Args:
#         scores (torch.Tensor): Predicted scores for n images, shape (n,).
#         true_ranks (torch.Tensor): Ground-truth ranks for n images, shape (n,).
#                                    Lower rank means better (0 = best).

#     Returns:
#         Scalar loss value (torch.Tensor).
#     """
#     # Get permutation indices by sorting true ranks in ascending order.
#     _, perm = torch.sort(true_ranks, descending=False)
#     sorted_scores = scores[perm]
#     n = sorted_scores.shape[0]
#     loss = 0.0
#     for i in range(n):
#         loss += - sorted_scores[i] + torch.log(torch.sum(torch.exp(sorted_scores[i:])))
#     return loss

def listmle_loss(scores, true_ranks):
    # Subtract the max score for stability
    scores = scores - scores.max()
    _, perm = torch.sort(true_ranks, descending=False)
    sorted_scores = scores[perm]
    reverse_cum_logsumexp = torch.flip(
        torch.logcumsumexp(torch.flip(sorted_scores, dims=(0,)), dim=0),
        dims=(0,)
    )
    loss = torch.sum(-sorted_scores + reverse_cum_logsumexp)
    return loss
# def listmle_loss(scores, true_ranks):
#     # ---- move to FP32 before any reductions that amp can’t handle ----
#     scores      = scores.float()               # FP32
#     true_ranks  = true_ranks.long()            # index tensor

#     scores = scores - scores.max()
#     _, perm = torch.sort(true_ranks, descending=False)
#     sorted_scores = scores[perm]

#     rev_cum_lse = torch.flip(
#         torch.logcumsumexp(torch.flip(sorted_scores, dims=(0,)), dim=0),
#         dims=(0,)
#     )
#     return torch.sum(-sorted_scores + rev_cum_lse)
# ##############################################################################
# Utility Functions for Evaluation & Visualization
##############################################################################
def inversion_score(p1, p2):
    """
    Computes inversion score between two rankings (1 indicates perfect ranking).
    """
    assert len(p1) == len(p2), f'{len(p1)}, {len(p2)}'
    n = len(p1)
    cnt = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (p1[i] > p1[j] and p2[i] < p2[j]) or (p1[i] < p1[j] and p2[i] > p2[j]):
                cnt += 1
    return 1 - cnt / (n * (n - 1) / 2)

def validate_sda(aggregator, model, dataloader, device):
    aggregator.eval()
    total_inversion = 0.0
    total_examples = 0
    all_rankings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images, num_images, labels, texts = batch
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)
            num_images = num_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images, texts)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            scale = outputs["logit_scale"]
            batch_scores = aggregator(image_features, text_features, num_images, scale)
            predicted = [torch.argsort(-scores) for scores in batch_scores]
            sda_rankings = [[pred.tolist().index(j) for j in range(n)]
                            for pred, n in zip(predicted, num_images)]
            true_rankings = [label for label in labels.split(num_images.tolist())]
            all_rankings.extend(sda_rankings)
            for i in range(len(sda_rankings)):
                total_inversion += inversion_score(sda_rankings[i], true_rankings[i])
                total_examples += 1
    final_score = total_inversion / total_examples if total_examples > 0 else 0.0
    print(f'Validation ranking inversion score: {final_score:.4f}')
    os.makedirs('logs', exist_ok=True)
    with open('logs/sda_rankings.json', 'w') as f:
        json.dump(all_rankings, f)
    return final_score


##########################################################################
# PART C: Training Pipeline
##########################################################################
def train_sda(aggregator, model, train_dataloader, val_dataloader, epochs=3, lr=1e-5, device="cpu"):
    aggregator.to(device)
    # optimizer = optim.AdamW(aggregator.parameters(), lr=lr, weight_decay=0.35)
    optimizer = optim.Adam(aggregator.parameters(), lr=lr, weight_decay=1e-5)
    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # e.g., 10% warmup
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler()
    pwh_loss = PairwiseHingeRankingLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience = 5
    no_improve = 0
    val_iterator = iter(val_dataloader)

    global_step = 0
    for epoch in range(epochs):
        aggregator.train()
        running_loss = 0.0
        running_acc = 0.0
        running_top1 = 0.0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, num_images, ranks, texts = batch
            images = images.to(device)
            texts = texts.to(device)
            num_images = num_images.to(device)
            ranks = ranks.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images, texts)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                scale = outputs["logit_scale"]

                batch_scores = aggregator(image_features, text_features, num_images, scale)

                loss = 0.0
                offset = 0
                for i, score in enumerate(batch_scores):
                    n_images = int(num_images[i].item()) if hasattr(num_images[i], "item") else num_images[i]
                    ranks_i = ranks[offset: offset + n_images]
                    
                    min_idx = torch.argmin(ranks_i)                 # index of the smallest element
                    one_hot = F.one_hot(min_idx, num_classes=ranks_i.size(0))
                    # one_hot is a LongTensor([1, 0, 0, 0])

                    # if you need floats:
                    one_hot = one_hot.float()
                    one_hot = F.softmax(-ranks_i.float(), dim=0)
                    # mask = score == score.max()        # → tensor([False, False,  True, False])

                    # # fill all masked‐out positions with –inf
                    # masked_logits = score.masked_fill(~mask, float("-inf"))
                    # print(score)
                    # print(masked_logits)
                    # loss += listmle_loss(score, ranks_i)
                    loss += cross_entropy_loss(score, one_hot)
                    # loss += pwh_loss(score, ranks_i)
                    offset += n_images
                loss = loss / len(batch)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)        
            torch.nn.utils.clip_grad_norm_(aggregator.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()  # Step scheduler at every batch

            # current_lr = scheduler.get_last_lr()[0]  # or optimizer.param_groups[0]['lr']
            
            # Optionally, compute batch accuracy etc. here...
            with torch.no_grad():
                batch_acc = 0.0
                batch_top1 = 0.0
                split_labels = ranks.split(num_images.tolist())
                for s, r in zip(batch_scores, split_labels):
                    batch_acc += inversion_score(torch.argsort(-s).tolist(), r.tolist())
                    batch_top1 += top1_accuracy(s, r)
                batch_acc = batch_acc / len(batch_scores)
                batch_top1 = batch_top1 / len(batch_scores)

            wandb.log({
                "train/loss": loss.item(),
                "train/ranking_accuracy": batch_acc,
                "train/top1_accuracy": batch_top1,
                "train/learning_rate": lr,
            }, step=global_step)
            global_step += 1

            running_loss += loss.item()
            running_acc += batch_acc
            running_top1 += batch_top1
            num_batches += 1

            if num_batches % 50 == 0:
                aggregator.eval()
                val_acc = validate_sda(aggregator, model, val_dataloader, device)
                wandb.log({
                    "epoch": epoch,
                    "batch": num_batches,
                    "val/accuracy": val_acc
                }, step=global_step)
                aggregator.train()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_improve = 0
                    torch.save(aggregator.state_dict(), 'mask_ce_hunch_bigger_lr_soft_labels.pt')

        avg_loss = running_loss / num_batches
        avg_acc = running_acc / num_batches
        avg_top1 = running_top1 / num_batches
        print(f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, Top1 Acc={avg_top1:.4f}")
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": avg_loss,
            "train/epoch_accuracy": avg_acc,
            "train/epoch_top1_accuracy": avg_top1
        }, step=global_step)

        val_acc = validate_sda(aggregator, model, val_dataloader, device)
        wandb.log({
            "epoch": epoch,
            "val/epoch_accuracy": val_acc
        }, step=global_step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(aggregator.state_dict(), 'mask_ce_hunch_bigger_lr_soft_labels.pt')
        # Optionally implement early stopping here

    print("Training complete.")

def kld(self, vec_true, vec_compare):
    ind = vec_true * vec_compare > 0
    include_nan = vec_true * torch.log(vec_true / vec_compare)
    z = torch.zeros_like(ind, dtype=torch.float32)
    return torch.sum(torch.where(ind, include_nan, z))

def jsd(self, vec_true, vec_compare):
    vec_mean = 0.5 * (vec_true + vec_compare)
    return 0.5 * self.kld(vec_true, vec_mean) + 0.5 * self.kld(vec_compare, vec_mean)

def topkprob(self, vec, k=5):
    # Sort the vector in descending order
    vec_sort, _ = torch.sort(vec, descending=True)
    topk = vec_sort[:k]
    ary = torch.arange(k, device=vec.device)
    
    # Compute the top-k probabilities
    topk_prob = torch.prod(torch.exp(topk) / torch.sum(torch.exp(topk[ary:]), dim=0))
    
    return topk_prob

def listwise_cost(self, list_ans, list_pred):
    return - torch.sum(self.topkprob(list_ans) * torch.log(self.topkprob(list_pred)))

##########################################################################
# PART C: Main training pipeline
##########################################################################
def main():
    set_seed(42)
    ranked_json_path = "/home/arkadi.piven/.cache/huggingface/datasets/downloads/extracted/151cbc6b3e6e668e5769dcee70ed385f0ccda7a828473238ee2f83cfced36f33/train/train.json"
    image_folder = "/home/arkadi.piven/.cache/huggingface/datasets/downloads/extracted/151cbc6b3e6e668e5769dcee70ed385f0ccda7a828473238ee2f83cfced36f33/train"
    with open(ranked_json_path, "r") as f:
        ranking_data = json.load(f)
    print("Loaded ranking data from:", ranked_json_path)
    print("First item:", ranking_data[0])

    model_name = "ViT-H-14"  # from --model ViT-H-14
    pretrained_ckpt = "laion2B-s32B-b79K"  # from --pretrained laion2B-s32B-b79K
    precision = 'amp'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model, preprocess_train, preprocess_val = create_model_and_transforms(
    #     model_name=model_name,
    #     pretrained=pretrained_ckpt,
    #     precision=precision,
    #     device=device,
    #     jit=False,
    #     force_quick_gelu=False,
    #     force_custom_text=False,
    #     force_patch_dropout=False,
    #     force_image_size=None,
    #     pretrained_image=False,
    #     image_mean=None,
    #     image_std=None,
    #     light_augmentation=True,
    #     aug_cfg={},
    #     output_dict=True,
    #     with_score_predictor=False,
    #     with_region_predictor=False
    # )

    # tokenizer = get_tokenizer(model_name)
    # model = model.to(device)
    # model.eval()

    model_dict = {}

    def initialize_model():
        if not model_dict:
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                model_name,
                None,
                precision=precision,
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_train'] = preprocess_train
            model_dict['preprocess_val'] = preprocess_val

    initialize_model()

    model = model_dict['model']
    preprocess_train = model_dict['preprocess_train']
    preprocess_val = model_dict['preprocess_val']

    print('Loading model ...')
    checkpoint = torch.load("/home/arkadi.piven/Code/HumanGuidedDiffusion/shelly/hpsv2/HPS_v2_compressed.pt", map_location="cuda")
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    # set_openclip_trainability(model, img_blocks_to_train=20, txt_blocks_to_train=11)

    for param in model.parameters():
        param.requires_grad = False

    # if hasattr(model.visual, 'transformer'):
    #     # Example: unfreeze the last two blocks
    #     for block_idx in [-2, -1]:
    #         for param in model.visual.transformer.resblocks[block_idx].parameters():
    #             param.requires_grad = True

    # # 5. Unfreeze the top blocks of the text Transformer
    # if hasattr(model, 'transformer'):
    #     # Example: unfreeze the last block only
    #     for param in model.transformer.resblocks[-1].parameters():c
    #         param.requires_grad = True

    # # 6. Unfreeze layer norms (often helps to adapt to new data distribution)
    # for name, param in model.named_parameters():
    #     # If the parameter is part of a layer norm, unfreeze it
    #     if "ln_" in name or "layer_norm" in name or "LayerNorm" in name:
    #         param.requires_grad = True

    # # 7. Double-check which layers are trainable
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)


    dataset = RankingDataset(ranked_json_path, image_folder, preprocess_train, tokenizer)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16, collate_fn=collate_rank, drop_last=True, pin_memory=True)

    meta_file = os.path.join("/home/arkadi.piven/.cache/huggingface/datasets/downloads/extracted/df7d02e5efe9db8a0270f42fa4ff8be19c3a5fda53ea41fd0616340fe0f93b2b/test/", 'test.json')
    dataset_test = RankingDataset(meta_file, "/home/arkadi.piven/.cache/huggingface/datasets/downloads/extracted/df7d02e5efe9db8a0270f42fa4ff8be19c3a5fda53ea41fd0616340fe0f93b2b/test/", preprocess_val, tokenizer)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16, collate_fn=collate_rank)

    aggregator = FinalSDA(1024,512)
    train_sda(aggregator, model, dataloader, dataloader_test, epochs=10, lr=1e-3, device=device)

if __name__ == "__main__":
    main()

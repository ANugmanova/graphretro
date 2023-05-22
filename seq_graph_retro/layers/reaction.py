import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

from seq_graph_retro.molgraph.mol_features import BINARY_FDIM


def create_scope_tensor(atom_scope, device=None):
    scope_sizes = [i[1] for i in atom_scope]
    return scope_sizes, scope_sizes


def flat_to_batch(c_atom, scope_tensor):
    c_atom_batch = torch.split(c_atom[1:], scope_tensor)
    c_atom_batch = nn.utils.rnn.pad_sequence(c_atom_batch, batch_first=True)
    return c_atom_batch


def batch_to_flat(c_atom_batch, scope_tensor):
    c_atom_flat = []
    for mol, size in zip(c_atom_batch, scope_tensor):
        c_atom_flat.append(mol[:size])
    c_atom_flat = torch.cat(c_atom_flat, dim=0)
    return c_atom_flat


def get_pair(c_atom_batch):
    max_atoms = c_atom_batch.shape[1]
    x_i = c_atom_batch.unsqueeze(2).expand(-1, -1, max_atoms, -1)
    x_j = c_atom_batch.unsqueeze(1).expand(-1, max_atoms, -1, -1)
    return x_i - x_j


class AtomAttention(nn.Module):
    """Pairwise atom attention layer."""
    # Update this attention layer

    def __init__(self,
                 n_bin_feat: int = BINARY_FDIM,
                 hsize: int = 64,
                 n_heads: int = 4,
                 bias: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        n_bin_feat: int, default BINARY_FDIM(11):
            Number of binary features used
        hsize: int, default 64
            Size of the embeddings
        n_heads: int, default 4
            Number of attention heads
        device: str, default cpu
            Device on which the programme is running
        bias: bool, default False
            Whether to use a bias term in the linear layers
        """
        super(AtomAttention, self).__init__(**kwargs)
        self.n_bin_feat = n_bin_feat
        self.hsize = hsize
        self.n_heads = n_heads
        self.bias = bias
        self._build_layer_components()

    def _build_layer_components(self) -> None:
        """Builds the different layers associated."""
        self.Wa_pair = nn.Linear(self.hsize, self.hsize * self.n_heads, self.bias)
        # self.Wa_bin = nn.Linear(self.n_bin_feat, self.hsize * self.n_heads, bias=True)
        self.Wa_score = nn.Parameter(torch.FloatTensor(self.hsize, 1, self.n_heads))
        self.W_proj = nn.Linear(self.hsize * self.n_heads, self.hsize, self.bias)
        nn.init.kaiming_uniform_(self.Wa_score, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor, scopes: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward propagation step.

        Parameters
        ----------
        inputs: torch.Tensor
            Atom embeddings from MPNN-Encoder
        scopes: Tuple[List]
            Scopes is composed of atom and bond scopes, which keep track of
            atom and bond indices for each molecule in the 2D feature list
        """
        c_atom = inputs
        atom_scope, bond_scope = scopes
        scope_tensor, scope_rev_tensor = create_scope_tensor(atom_scope, device=c_atom.device)
        c_atom_batch = flat_to_batch(c_atom, scope_tensor)
        atom_pair = get_pair(c_atom_batch)

        bs, max_atoms = atom_pair.size(0), atom_pair.size(1)
        target_shape = [bs, max_atoms, max_atoms, self.hsize, self.n_heads]

        pair_att = self.Wa_pair(atom_pair)
        total_att = F.relu(pair_att).view(target_shape)
        assert list(total_att.shape) == target_shape

        eq = '...hn,hjn->...jn'
        att_score = torch.sigmoid(torch.einsum(eq, [total_att, self.Wa_score]))
        #att_score = att_score * attn_mask.unsqueeze(-1).unsqueeze(-1) # Mask deals with dummy atoms

        c_atom_exp = c_atom_batch.unsqueeze(1).unsqueeze(-1)
        c_atom_att = att_score * c_atom_exp
        c_atom_att = self.W_proj(torch.sum(c_atom_att, dim=2).view(bs, max_atoms, -1))
        assert list(c_atom_att.shape) == [bs, max_atoms, self.hsize]
        c_mol_att = c_atom_att.sum(dim=1)
        c_atom_att = batch_to_flat(c_atom_att, scope_rev_tensor)

        c_atom_att = torch.cat([c_atom_att.new_zeros(1, self.hsize), c_atom_att], dim=0)
        return c_mol_att, c_atom_att


class PairFeat(nn.Module):
    """Computes embeddings for pairs of atoms. Precursor to predicting bond formation."""

    def __init__(self,
                 n_bin_feat: int = BINARY_FDIM,
                 hsize: int = 64,
                 n_heads: int = 4,
                 bias: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        n_bin_feat: int, default BINARY_FDIM(11):
            Number of binary features used
        hsize: int, default 64
            Size of the embeddings
        n_heads: int, default 4,
            Number of attention heads
        bias: bool, default False
            Whether to use bias in linear layers
        """
        super(PairFeat, self).__init__(**kwargs)
        self.n_bin_feat = n_bin_feat
        self.hsize = hsize
        self.bias = bias
        self._build_layer_components()

    def _build_layer_components(self) -> None:
        """Builds layer components."""
        self.Wp_a_pair = nn.Linear(self.hsize, self.hsize, self.bias)
        self.Wp_att_pair = nn.Linear(self.hsize, self.hsize, self.bias)
        self.Wp_bin = nn.Linear(self.n_bin_feat, self.hsize, self.bias)

    def forward(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs: Tuple[torch.Tensor]
            Inputs for pair feat computation
        """
        atom_pair, c_atom_att, bin_feat = inputs
        atom_att_pair = get_pair(c_atom_att)
        pair_hidden = self.Wp_a_pair(atom_pair) + self.Wp_att_pair(atom_att_pair) + \
                      self.Wp_bin(bin_feat)
        pair_hidden = F.relu(pair_hidden)
        return pair_hidden

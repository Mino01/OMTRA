import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from typing import Union, Callable, Dict, Optional
import scipy
from typing import List
from omtra.models.gvp import HeteroGVPConv, GVP, _norm_no_nan, _rbf
from omtra.utils.embedding import get_time_embedding
from omtra.data.graph import to_canonical_etype
from omtra.constants import (
    lig_atom_type_map,
    npnde_atom_type_map,
    ph_idx_to_type,
    residue_map,
    protein_element_map,
    protein_atom_map,
)


class EndpointVectorField(nn.Module):
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[str],
        n_atom_types: int,
        canonical_feat_order: list,
        interpolant_scheduler: InterpolantScheduler,
        n_charges: int = 6,
        n_bond_types: int = 4,
        n_vec_channels: int = 16,
        n_cp_feats: int = 0,
        n_hidden_scalars: int = 64,
        n_hidden_edge_feats: int = 64,
        n_recycles: int = 1,
        n_molecule_updates: int = 2,
        convs_per_update: int = 2,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        n_expansion_gvps: int = 3,
        separate_mol_updaters: bool = False,
        message_norm: Union[float, str] = 100,
        update_edge_w_distance: bool = False,
        rbf_dmax=20,
        rbf_dim=16,
        continuous_inv_temp_schedule=None,
        continuous_inv_temp_max: float = 10.0,
        time_embedding_dim: int = 64,
        token_dim: int = 64,
        attention: bool = False,
        n_heads: int = 1,
        s_message_dim: int = None,
        v_message_dim: int = None,
        dropout: float = 0.0,
        has_mask: bool = True,
        self_conditioning: bool = False,
        use_dst_feats: bool = False,
        dst_feat_msg_reduction_factor: float = 4,
        # if we are using CTMC, input categorical features will have mask tokens,
        # this means their one-hot representations will have an extra dimension,
        # and the neural network instantiated by this method need to account for this
        # it is definitely anti-pattern to have a parameter in parent class that is only needed for one sub-class (CTMCVectorField)
        # however, this is the fastest way to get CTMCVectorField working right now, so we will be anti-pattern for the sake of time
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.token_dim = token_dim
        self.n_lig_atom_types = len(lig_atom_type_map)
        self.n_npnde_atom_types = len(npnde_atom_type_map)
        self.n_protein_atom_types = len(protein_atom_map)
        self.n_protein_residue_types = len(residue_map)
        self.n_protein_element_types = len(protein_element_map)
        self.n_pharm_types = len(ph_idx_to_type)
        self.n_cross_edge_types = (
            2  # NOTE: un-hard code eventually (2 for proximity, covalent)
        )
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters: bool = separate_mol_updaters
        self.interpolant_scheduler = interpolant_scheduler
        self.canonical_feat_order = canonical_feat_order
        self.time_embedding_dim = time_embedding_dim
        self.self_conditioning = self_conditioning
        self.has_mask = has_mask

        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        assert n_vec_channels >= 3, "n_vec_channels must be >= 3"

        self.continuous_inv_temp_schedule = continuous_inv_temp_schedule
        self.continouts_inv_temp_max = continuous_inv_temp_max
        self.continuous_inv_temp_func = self.build_continuous_inv_temp_func(
            self.continuous_inv_temp_schedule, self.continouts_inv_temp_max
        )

        self.n_cat_feats = {  # number of possible values for each categorical variable (not including mask tokens in the case of CTMC)
            "lig_a": self.n_lig_atom_types,
            "lig_c": self.n_charges,
            "lig_e": self.n_bond_types,
            "npnde_a": self.n_npnde_atom_types,
            "npnde_c": self.n_charges,
            "npnde_e": self.n_bond_types,
            "pharm_a": self.n_pharm_types,
            "pharm_c": 0,
            "pharm_e": 0,
            "prot_atom_a": self.n_protein_atom_types,
            "prot_atom_e": self.n_protein_element_types,
            "prot_atom_edge": self.n_cross_edge_types,
            "prot_atom_r": self.n_protein_residue_types,
        }

        self.token_dims = {
            cat_feat: token_dim
            for cat_feat, n_unique in self.n_cat_feats.items()
            if n_unique > 0
        }

        mask_feats = {"lig_a", "lig_c", "lig_e", "pharm_a"}
        n_mask_feats = int(has_mask)

        # create token embeddings as identity layers
        # for non-CTMC parameterizations we need to repersent
        # categorical features as continuous vectors
        # but when we use CTMC we can use actual embedding functions
        self.token_embeddings = nn.ModuleDict()
        for feat, n_unique in self.n_cat_feats.items():
            if token_dim == 0:
                self.token_embeddings[feat] = nn.Identity()
            else:
                count = n_unique + n_mask_feats if feat in mask_feats else n_unique
                self.token_embeddings[feat] = nn.Embedding(count, token_dim)

        # fix 0 token dims to the number of categories
        for modality, token_dim in self.token_dims.items():
            if token_dim == 0:
                self.token_dims[modality] = (
                    self.n_cat_feats[modality] + n_mask_feats
                    if modality in mask_feats
                    else self.n_cat_feats[modality]
                )

        self.edge_feat_sizes = {}
        for etype in self.edge_types:
            if etype in {"lig_lig", "prot_atom_to_lig"}:
                self.edge_feat_sizes[etype] = n_hidden_edge_feats
            else:
                self.edge_feat_sizes[etype] = 0

        self.scalar_embedding = nn.ModuleDict()
        self.edge_embedding = nn.ModuleDict()

        for ntype in self.node_types:
            if ntype in ["lig", "npnde", "pharm"]:
                self.scalar_embedding[ntype] = nn.Sequential(
                    nn.Linear(
                        self.token_dims[f"{ntype}_a"]
                        + self.token_dims[f"{ntype}_c"]
                        + self.time_embedding_dim,
                        n_hidden_scalars,
                    ),
                    nn.SiLU(),
                    nn.Linear(n_hidden_scalars, n_hidden_scalars),
                    nn.SiLU(),
                    nn.LayerNorm(n_hidden_scalars),
                )
                if ntype == ["lig", "npnde"]:
                    self.edge_embedding[ntype] = nn.Sequential(
                        nn.Linear(self.token_dims[f"{ntype}_e"], n_hidden_edge_feats),
                        nn.SiLU(),
                        nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
                        nn.SiLU(),
                        nn.LayerNorm(n_hidden_edge_feats),
                    )
            elif ntype == "prot_atom":
                self.scalar_embedding[ntype] = nn.Sequential(
                    nn.Linear(
                        self.token_dims[f"{ntype}_a"]
                        + self.token_dims[f"{ntype}_e"]
                        + self.token_dims[f"{ntype}_r"]
                        + self.time_embedding_dim,
                        n_hidden_scalars,
                    ),
                    nn.SiLU(),
                    nn.Linear(n_hidden_scalars, n_hidden_scalars),
                    nn.SiLU(),
                    nn.LayerNorm(n_hidden_scalars),
                )
                self.edge_embedding["cross"] = nn.Sequential(
                    nn.Linear(self.token_dims[f"{ntype}_edge"], n_hidden_edge_feats),
                    nn.SiLU(),
                    nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
                    nn.SiLU(),
                    nn.LayerNorm(n_hidden_edge_feats),
                )
        for etype in self.edge_types:
            if self.edge_feat_sizes[etype] > 0:
                self.edge_embedding[etype] = nn.Sequential(
                    nn.Linear(token_dim, n_hidden_edge_feats),
                    nn.SiLU(),
                    nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
                    nn.SiLU(),
                    nn.LayerNorm(n_hidden_edge_feats),
                )

        self.conv_layers = []
        for conv_idx in range(convs_per_update * n_molecule_updates):
            self.conv_layers.append(
                HeteroGVPConv(
                    node_types=self.node_types,
                    edge_types=self.edge_types,
                    scalar_size=n_hidden_scalars,
                    vector_size=n_vec_channels,
                    n_cp_feats=n_cp_feats,
                    edge_feat_size=self.edge_feat_sizes,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    n_expansion_gvps=n_expansion_gvps,
                    message_norm=message_norm,
                    rbf_dmax=rbf_dmax,
                    rbf_dim=rbf_dim,
                    attention=attention,
                    n_heads=n_heads,
                    s_message_dim=s_message_dim,
                    v_message_dim=v_message_dim,
                    dropout=dropout,
                    use_dst_feats=use_dst_feats,
                    dst_feat_msg_reduction_factor=dst_feat_msg_reduction_factor,
                )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # create molecule update layers
        self.node_position_updaters = nn.ModuleDict()
        self.edge_updaters = nn.ModuleDict()
        if self.separate_mol_updaters:
            n_updaters = n_molecule_updates
        else:
            n_updaters = 1
        for ntype in self.node_types:
            self.node_position_updaters[ntype] = nn.ModuleList()
            for _ in range(n_updaters):
                self.node_position_updaters[ntype].append(
                    NodePositionUpdate(
                        n_hidden_scalars,
                        n_vec_channels,
                        n_gvps=3,
                        n_cp_feats=n_cp_feats,
                    )
                )

        for etype in self.edge_types:
            if self.edge_feat_sizes[etype] > 0:
                self.edge_updaters[etype] = nn.ModuleList()
                for _ in range(n_updaters):
                    self.edge_updaters[etype].append(
                        EdgeUpdate(
                            self.node_types,
                            self.edge_types,
                            n_hidden_scalars,
                            n_hidden_edge_feats,
                            update_edge_w_distance=update_edge_w_distance,
                            rbf_dim=rbf_dim,
                        )
                    )

        self.node_output_heads = nn.ModuleDict()
        for ntype in ["lig", "pharm"]:
            output_dim = 0
            if ntype == "lig":
                output_dim = self.n_lig_atom_types + n_charges
            elif ntype == "pharm":
                output_dim = self.n_pharm_types
            self.node_output_heads[ntype] = nn.Sequential(
                nn.Linear(n_hidden_scalars, n_hidden_scalars),
                nn.SiLU(),
                nn.Linear(n_hidden_scalars, output_dim),
            )

        self.edge_output_heads = nn.ModuleDict()
        for etype in ["lig_lig", "prot_atom_to_lig"]:
            output_dim = n_bond_types
            if etype == "prot_atom_to_lig":
                output_dim = self.n_cross_edge_types
            self.edge_output_heads[etype] = nn.Sequential(
                nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
                nn.SiLU(),
                nn.Linear(n_hidden_edge_feats, output_dim),
            )

        if self.self_conditioning:
            raise NotImplementedError("Self conditioning not implemented yet")
            self.self_conditioning_residual_layer = SelfConditioningResidualLayer(
                n_atom_types=n_atom_types,
                n_charges=n_charges,
                n_bond_types=n_bond_types,
                node_embedding_dim=n_hidden_scalars,
                edge_embedding_dim=n_hidden_edge_feats,
                rbf_dim=rbf_dim,
                rbf_dmax=rbf_dmax,
            )

    def build_continuous_inv_temp_func(self, schedule, max_inv_temp=None):
        if schedule is None:
            inv_temp_func = lambda t: 1.0
        elif schedule == "linear":
            inv_temp_func = lambda t: max_inv_temp * (1 - t)
        elif callable(schedule):
            inv_temp_func = schedule
        else:
            raise ValueError(f"Invalid continuous_inv_temp_schedule: {schedule}")
        return inv_temp_func

    def forward(
        self,
        g: dgl.DGLGraph,
        t: torch.Tensor,
        node_batch_idx: Dict[str, torch.Tensor],
        upper_edge_mask: torch.Tensor,
        apply_softmax=False,
        remove_com=False,
        prev_dst_dict=None,
    ):
        """Predict x_1 (trajectory destination) given x_t, and, optionally, previous destination features."""
        device = g.device

        with g.local_scope():
            node_scalar_features = {}
            node_positions = {}
            node_vec_features = {}
            edge_features = {}

            for ntype in self.node_types:
                node_positions[ntype] = g.nodes[ntype].data["x_t"]
                num_nodes = g.num_nodes(ntype)
                node_vec_features[ntype] = torch.zeros(
                    (num_nodes, self.n_vec_channels, 3), device=device
                )
                scalar_feats = []
                if (
                    f"{ntype}_a" in self.token_embeddings
                    and self.token_embeddings[f"{ntype}_a"] is not None
                ):
                    scalar_feats.append(
                        self.token_embeddings[f"{ntype}_a"](
                            g.nodes[ntype].data["a_t"].argmax(dim=-1)
                        )
                    )
                else:
                    scalar_feats.append(g.nodes[ntype].data["a_t"])
                if ntype in ["lig", "npnde"]:
                    if (
                        f"{ntype}_c" in self.token_embeddings
                        and self.token_embeddings[f"{ntype}_c"] is not None
                    ):
                        scalar_feats.append(
                            self.token_embeddings[f"{ntype}_c"](
                                g.nodes[ntype].data["c_t"].argmax(dim=-1)
                            )
                        )
                    else:
                        scalar_feats.append(g.nodes[ntype].data["c_t"])
                if ntype == "prot_atom":
                    if (
                        f"{ntype}_r" in self.token_embeddings
                        and self.token_embeddings[f"{ntype}_r"] is not None
                    ):
                        scalar_feats.append(
                            self.token_embeddings[f"{ntype}_r"](
                                g.nodes[ntype].data["r_t"].argmax(dim=-1)
                            )
                        )
                    else:
                        scalar_feats.append(g.nodes[ntype].data["r_t"])

                    if (
                        f"{ntype}_e" in self.token_embeddings
                        and self.token_embeddings[f"{ntype}_e"] is not None
                    ):
                        scalar_feats.append(
                            self.token_embeddings[f"{ntype}_e"](
                                g.nodes[ntype].data["e_t"].argmax(dim=-1)
                            )
                        )
                    else:
                        scalar_feats.append(g.nodes[ntype].data["e_t"])

                if self.time_embedding_dim == 1:
                    scalar_feats.append(t[node_batch_idx[ntype]].unsqueeze(-1))
                else:
                    t_emb = get_time_embedding(t, embedding_dim=self.time_embedding_dim)
                    t_emb = t_emb[node_batch_idx[ntype]]
                    scalar_feats.append(t_emb)

                scalar_feat = torch.cat(scalar_feats, dim=-1)
                node_scalar_features[ntype] = self.scalar_embedding[ntype](scalar_feat)

            for etype in self.edge_types:
                if self.edge_feat_sizes[etype] > 0:
                    if etype in {"lig_to_lig", "prot_atom_to_lig"}:
                        edge_feats = g.edges(etype=etype).data["e_t"]
                        edge_feats = self.edge_embedding[etype](edge_feats)
                        edge_features[etype] = edge_feats

            if self.self_conditioning and prev_dst_dict is None:
                train_self_condition = self.training and (torch.rand(1) > 0.5).item()
                inference_first_step = not self.training and (t == 0).all().item()

                if train_self_condition or inference_first_step:
                    with torch.no_grad():
                        # Clone features for gradient-stopped pass
                        node_scalar_features_clone = {
                            ntype: feats.clone()
                            for ntype, feats in node_scalar_features.items()
                        }
                        node_vec_features_clone = {
                            ntype: feats.clone()
                            for ntype, feats in node_vec_features.items()
                        }
                        node_positions_clone = {
                            ntype: pos.clone() for ntype, pos in node_positions.items()
                        }
                        edge_features_clone = {
                            etype: feats.clone()
                            for etype, feats in edge_features.items()
                        }

                        prev_dst_dict = self.denoise_graph(
                            g,
                            node_scalar_features_clone,
                            node_vec_features_clone,
                            node_positions_clone,
                            edge_features_clone,
                            node_batch_idx,
                            upper_edge_mask,
                            apply_softmax=True,
                            remove_com=False,
                        )

            if self.self_conditioning and prev_dst_dict is not None:
                # TODO: Adapt self-conditioning residual layer for Hetero graph
                (
                    node_scalar_features,
                    node_positions,
                    node_vec_features,
                    edge_features,
                ) = self.self_conditioning_residual_layer(
                    g,
                    node_scalar_features,
                    node_positions,
                    node_vec_features,
                    edge_features,
                    prev_dst_dict,
                    node_batch_idx,
                    upper_edge_mask,
                )

            dst_dict = self.denoise_graph(
                g,
                node_scalar_features,
                node_vec_features,
                node_positions,
                edge_features,
                node_batch_idx,
                upper_edge_mask,
                apply_softmax,
                remove_com,
            )

            return dst_dict

        def denoise_graph(
            self,
            g: dgl.DGLGraph,
            node_scalar_features: Dict[str, torch.Tensor],
            node_vec_features: Dict[str, torch.Tensor],
            node_positions: Dict[str, torch.Tensor],
            edge_features: Dict[str, torch.Tensor],
            node_batch_idx: Dict[str, torch.Tensor],
            upper_edge_mask: torch.Tensor,
            apply_softmax: bool = False,
            remove_com: bool = False,
        ):
            # TODO: adapt flowmol denoise_graph for hetero version
            pass

        def precompute_distances(self, g: dgl.DGLGraph, node_positions=None):
            # TODO: adapt flowmol precompute_distances for hetero version
            pass

        def integrate(self, g: dgl.DGLGraph):
            # TODO: adapt flowmol integrate for hetero version
            pass

        def step(self, g: dgl.DGLGraph):
            # TODO: adapt flowmol step for hetero version
            pass

        def vector_field(self, x_t, x_1, alpha_t, alpha_t_prime):
            vf = alpha_t_prime / (1 - alpha_t) * (x_1 - x_t)
            return vf


class NodePositionUpdate(nn.Module):
    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):
            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(
        self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor
    ):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)


class EdgeUpdate(nn.Module):
    def __init__(
        self,
        node_types,
        edge_types,
        n_node_scalars,
        n_edge_feats,
        update_edge_w_distance=False,
        rbf_dim=16,
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars * 2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fns = nn.ModuleDict()
        for etype in self.edge_types:
            self.edge_update_fns[etype] = nn.Sequential(
                nn.Linear(input_dim, n_edge_feats),
                nn.SiLU(),
                nn.Linear(n_edge_feats, n_edge_feats),
                nn.SiLU(),
            )
        self.edge_norms = nn.ModuleDict()
        for etype in self.edge_types:
            self.edge_norms[etype] = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d, etype):
        src_ntype, _, dst_ntype = to_canonical_etype(etype)
        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges(etype=etype)

        mlp_inputs = [
            node_scalars[src_ntype][src_idxs],
            node_scalars[dst_ntype][dst_idxs],
            edge_feats,
        ]

        if self.update_edge_w_distance and d is not None:
            mlp_inputs.append(d)

        edge_feats = self.edge_norms[etype](
            edge_feats + self.edge_update_fns[etype](torch.cat(mlp_inputs, dim=-1))
        )
        return edge_feats

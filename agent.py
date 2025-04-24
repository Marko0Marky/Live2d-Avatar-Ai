# --- START OF FILE agent.py ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import logging
import sys
import math
import os
from typing import Optional, Dict, Tuple, List, Deque, Union, Any

# --- Config & Logger First ---
from config import MasterConfig as Config, DEVICE, logger

# --- PyTorch Geometric Imports ---
try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data, Batch
    from torch_scatter import scatter_mean # Use scatter_mean for aggregation
    PYG_AVAILABLE = True
    logger.info("PyTorch Geometric and torch_scatter found.")
except ImportError:
    logger.critical("PyTorch Geometric (PyG) or torch_scatter NOT FOUND. Agent requires PyG.")
    PYG_AVAILABLE = False
    sys.exit(1)

# --- Other Imports ---
from config import AGENT_SAVE_PATH, GPT_SAVE_PATH, OPTIMIZER_SAVE_PATH, TARGET_NET_SAVE_SUFFIX
from utils import MetronicLattice, MetaCognitiveMemory, is_safe, Experience # Uses simplified Experience
from ai_modules import EmotionalModule, TransformerGPT

# --- Agent Forward Return Type ---
AgentForwardReturnType = namedtuple('AgentForwardReturnType', [
    'emotions', 'belief_embedding', 'integration_I', 'rho_score', 'zeta',
    'feedback', 'stability', 'full_state', 'gamma', 'R', 'att_score',
    'self_consistency', 'tau_t', 'metric_g', 'value_pred', 'lyapunov_max_proxy'
])

class ConsciousAgent(nn.Module):
    def __init__(self, state_dim: int = Config.Agent.STATE_DIM, hidden_dim: int = Config.Agent.HIDDEN_DIM):
        super().__init__()
        if state_dim != 12: logger.critical(f"Agent requires STATE_DIM=12."); sys.exit(1)
        if not PYG_AVAILABLE: logger.critical("PyG must be installed."); sys.exit(1)

        self.state_dim = state_dim; self.hidden_dim = hidden_dim;
        self.emotion_dim = Config.Agent.EMOTION_DIM; self.qualia_dim = self.state_dim - self.emotion_dim
        self.gnn_hidden_dim = Config.Agent.GNN.GNN_HIDDEN_DIM
        if self.gnn_hidden_dim != self.hidden_dim:
             logger.warning(f"Agent HIDDEN_DIM ({self.hidden_dim}) != GNN_HIDDEN_DIM ({self.gnn_hidden_dim}). Using GNN dim.")
             self.hidden_dim = self.gnn_hidden_dim # Ensure consistency

        # --- Online Networks ---
        self.lattice = MetronicLattice(dim=self.state_dim, tau=Config.Agent.TAU)
        self.encoder = nn.Linear(self.state_dim, self.gnn_hidden_dim).to(DEVICE)
        self.emotional_module = EmotionalModule(input_dim=self.emotion_dim + 1).to(DEVICE)
        self.gnn_layers_module = self._build_gnn_module().to(DEVICE)
        self.self_reflect_layer = nn.Linear(self.gnn_hidden_dim, self.hidden_dim).to(DEVICE)
        self.qualia_output_head = nn.Linear(self.hidden_dim, self.qualia_dim).to(DEVICE)
        self.feedback = nn.Linear(self.hidden_dim, self.state_dim).to(DEVICE)
        self.value_head = nn.Linear(self.hidden_dim, 1).to(DEVICE)

        # --- Target Networks ---
        self.target_encoder = nn.Linear(self.state_dim, self.gnn_hidden_dim).to(DEVICE)
        self.target_gnn_layers_module = self._build_gnn_module().to(DEVICE)
        self.target_self_reflect_layer = nn.Linear(self.gnn_hidden_dim, self.hidden_dim).to(DEVICE)
        self.target_value_head = nn.Linear(self.hidden_dim, 1).to(DEVICE)

        self.update_target_network()
        self.set_target_networks_eval()
        self.soft_update_tau = Config.RL.TARGET_NETWORK_SOFT_UPDATE_TAU

        try: self.gpt = TransformerGPT()
        except Exception as e: logger.critical(f"GPT Init Error: {e}."); sys.exit(1)
        self.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE)

        online_params = list(self.encoder.parameters()) + list(self.gnn_layers_module.parameters()) + \
                        list(self.emotional_module.parameters()) + list(self.self_reflect_layer.parameters()) + \
                        list(self.feedback.parameters()) + list(self.qualia_output_head.parameters()) + \
                        list(self.value_head.parameters())
        self.optimizer = optim.AdamW([p for p in online_params if p.requires_grad], lr=Config.RL.LR, weight_decay=0.001)

        # State Tracking
        self.state_history_deque: Deque[torch.Tensor] = deque(maxlen=Config.Agent.HISTORY_SIZE)
        for _ in range(Config.Agent.HISTORY_SIZE): self.state_history_deque.append(torch.zeros(self.state_dim, device=DEVICE))
        self.prev_emotions = torch.zeros(self.emotion_dim, device=DEVICE)
        self.prev_belief_embedding: Optional[torch.Tensor] = None
        self.selector = torch.tensor(1.0, device=DEVICE); self.step_count = 0; self.beta = Config.RL.PER_BETA_START
        beta_frames = Config.RL.PER_BETA_FRAMES; self.beta_increment = (1.0 - self.beta) / max(1, beta_frames)
        self.entropy_proxy = torch.tensor(0.0, device=DEVICE)

        logger.info(f"ConsciousAgent initialized with PyG {Config.Agent.GNN.GNN_TYPE} layers and Target Networks.")

    def _create_gnn_layer(self, in_channels, out_channels):
        """Creates a PyG GNN layer based on config."""
        gnn_type=Config.Agent.GNN.GNN_TYPE.upper(); heads=Config.Agent.GNN.GAT_HEADS
        dropout_rate = 0.1
        if gnn_type=='GCN': layer = pyg_nn.GCNConv(in_channels, out_channels)
        elif gnn_type=='GAT': layer = pyg_nn.GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout_rate)
        elif gnn_type=='GRAPHSAGE': layer = pyg_nn.SAGEConv(in_channels, out_channels)
        elif gnn_type=='GIN': mlp=nn.Sequential(nn.Linear(in_channels, out_channels*2), nn.ReLU(), nn.Linear(out_channels*2, out_channels)).to(DEVICE); layer = pyg_nn.GINConv(mlp)
        else: logger.warning(f"Unknown GNN_TYPE '{gnn_type}'. Using Linear fallback."); layer = nn.Linear(in_channels, out_channels)
        return layer.to(DEVICE)

    def _build_gnn_module(self) -> nn.ModuleList:
        """Builds the stack of GNN layers."""
        gnn_layers = nn.ModuleList()
        current_channels = self.gnn_hidden_dim
        for i in range(Config.Agent.GNN.GNN_LAYERS):
            layer = self._create_gnn_layer(current_channels, self.gnn_hidden_dim)
            norm = nn.LayerNorm(self.gnn_hidden_dim).to(DEVICE)
            gnn_layers.append(nn.ModuleDict({'conv': layer, 'norm': norm}))
            current_channels = self.gnn_hidden_dim
        return gnn_layers

    def set_target_networks_eval(self):
        """Sets all target networks to evaluation mode."""
        self.target_encoder.eval()
        self.target_gnn_layers_module.eval()
        self.target_self_reflect_layer.eval()
        self.target_value_head.eval()

    def update_target_network(self):
        """Hard copies weights from online networks to target networks."""
        logger.debug("Hard updating ALL target networks...")
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_gnn_layers_module.load_state_dict(self.gnn_layers_module.state_dict())
        self.target_self_reflect_layer.load_state_dict(self.self_reflect_layer.state_dict())
        self.target_value_head.load_state_dict(self.value_head.state_dict())

    def _polyak_update(self, base_net, target_net, tau):
        """Performs Polyak averaging for soft target updates."""
        if base_net is None or target_net is None: return
        with torch.no_grad():
             if isinstance(base_net, nn.ModuleList) and isinstance(target_net, nn.ModuleList):
                 for target_module_dict, base_module_dict in zip(target_net, base_net):
                     for key in base_module_dict:
                         if isinstance(base_module_dict[key], nn.Module):
                             for target_p, base_p in zip(target_module_dict[key].parameters(), base_module_dict[key].parameters()):
                                  target_p.data.mul_(1.0 - tau); target_p.data.add_(tau * base_p.data)
             elif isinstance(base_net, nn.Module) and isinstance(target_net, nn.Module):
                 for target_p, base_p in zip(target_net.parameters(), base_net.parameters()):
                      target_p.data.mul_(1.0 - tau); target_p.data.add_(tau * base_p.data)

    def soft_update_target_networks(self, tau=None):
        """Soft updates all target networks."""
        update_tau = tau if tau is not None else self.soft_update_tau
        # logger.debug(f"Soft updating targets (tau={update_tau:.4f})") # Can be noisy
        self._polyak_update(self.encoder, self.target_encoder, update_tau)
        self._polyak_update(self.gnn_layers_module, self.target_gnn_layers_module, update_tau)
        self._polyak_update(self.self_reflect_layer, self.target_self_reflect_layer, update_tau)
        self._polyak_update(self.value_head, self.target_value_head, update_tau)

    @property
    def state_history(self) -> torch.Tensor:
        """Returns the 12D state history as a stacked tensor."""
        if not self.state_history_deque: return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)
        valid = all(isinstance(t, torch.Tensor) and t.shape == (self.state_dim,) for t in self.state_history_deque)
        # Ensure deque has enough elements before stacking
        if not valid or len(self.state_history_deque) < Config.Agent.HISTORY_SIZE:
             logger.warning(f"Agent history invalid/incomplete ({len(self.state_history_deque)}/{Config.Agent.HISTORY_SIZE}). Reinitializing.");
             self.state_history_deque.clear(); initial_state = torch.zeros(self.state_dim, device=DEVICE)
             for _ in range(Config.Agent.HISTORY_SIZE): self.state_history_deque.append(initial_state.clone())
        try: return torch.stack(list(self.state_history_deque)).to(DEVICE).float()
        except Exception as e: logger.error(f"Error stacking agent history: {e}."); return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)

    # === Helper Methods (Implementations using PyG/Batches) ===

    def build_graph(self, state_embedding: torch.Tensor) -> Data:
        """ Builds a PyG Data object for a single state embedding (fixed graph). """
        num_nodes = 7; node_features = state_embedding.unsqueeze(0).repeat(num_nodes, 1)
        edge_list = [[0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 6], [4, 6]];
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=DEVICE).t().contiguous()
        edge_attr = None # Default
        if Config.Agent.GNN.GNN_TYPE == 'GAT': # GAT needs edge attributes per head
             edge_attr = torch.ones(edge_index.shape[1], Config.Agent.GNN.GAT_HEADS, device=DEVICE) * 0.1 # Example small value
        return Data(x=node_features.float(), edge_index=edge_index, edge_attr=edge_attr)

    def run_gnn(self, graph_batch: Batch, use_target: bool = False) -> torch.Tensor:
        """ Processes a PyG Batch object through GNN layers. """
        layers = self.target_gnn_layers_module if use_target else self.gnn_layers_module
        x, edge_index, batch_map = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        edge_attr = getattr(graph_batch, 'edge_attr', None)
        if not is_safe(x): logger.warning("Unsafe input to run_gnn"); x = torch.zeros_like(x)
        try:
            for i, layer_block in enumerate(layers):
                x_res = x
                conv = layer_block['conv']; norm = layer_block['norm']
                # Pass edge_attr only if it exists and the layer type might use it
                if edge_attr is not None and isinstance(conv, (pyg_nn.GATConv, pyg_nn.TransformerConv)): # Add other types if needed
                    x = conv(x, edge_index, edge_attr=edge_attr)
                elif isinstance(conv, pyg_nn.MessagePassing): # Handles most PyG conv layers
                    x = conv(x, edge_index)
                elif isinstance(conv, nn.Linear): # Handle Linear fallback
                    x = conv(x) # Operates node-wise
                else: logger.error(f"Unsupported GNN layer type: {type(conv)}"); continue
                x = F.relu(x)
                x = norm(x)
                if Config.Agent.GNN.GNN_USE_RESIDUAL and x.shape == x_res.shape: x = x + x_res
                if not is_safe(x): raise ValueError(f"Unsafe output GNN layer {i}")
            aggregated_embedding = scatter_mean(x, batch_map, dim=0) # Aggregate node features per graph
            return aggregated_embedding if is_safe(aggregated_embedding) else torch.zeros(graph_batch.num_graphs, self.gnn_hidden_dim, device=DEVICE)
        except Exception as e: logger.error(f"GNN execution error: {e}", exc_info=True); return torch.zeros(graph_batch.num_graphs, self.gnn_hidden_dim, device=DEVICE)

    def compute_metric_batch(self, belief_embedding_batch: torch.Tensor) -> torch.Tensor:
        """ Computes the metric tensor proxy g_ik for a batch. """
        batch_size, hidden_dim = belief_embedding_batch.shape
        g_batch = torch.eye(hidden_dim, device=DEVICE, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
        try:
            belief_float = belief_embedding_batch.float()
            if not is_safe(belief_float): raise ValueError("Unsafe input belief embedding")
            belief_float = torch.nan_to_num(belief_float)
            g_plus = torch.einsum('bi,bj->bij', belief_float, belief_float)
            g_minus = torch.zeros_like(g_plus); g_real = g_plus; g_imag = g_minus
            stabilized_g_real_list = []
            current_selector = self.selector.detach() # Keep as tensor
            for i in range(batch_size):
                g_real_i = g_real[i].clone(); max_eig_i = torch.tensor(1.0, device=DEVICE)
                try:
                    if torch.isfinite(g_real_i).all():
                        eigvals_i = torch.linalg.eigvals(g_real_i).abs()
                        if eigvals_i.numel() > 0 and torch.isfinite(eigvals_i).all(): max_eig_i = eigvals_i.max()
                        else: max_eig_i = torch.tensor(100.0, device=DEVICE)
                    else: max_eig_i = torch.tensor(100.0, device=DEVICE)
                except torch.linalg.LinAlgError: max_eig_i = torch.tensor(100.0, device=DEVICE)
                except Exception as eig_e : logger.warning(f"Metric Eig Err sample {i}: {eig_e}"); max_eig_i = torch.tensor(100.0, device=DEVICE)
                if max_eig_i > current_selector:
                    stabilization_factor = (current_selector / max_eig_i.clamp(min=1e-6))
                    stabilization_factor_clamped = stabilization_factor.clamp(min=0.01, max=1.0)
                    g_real_i = g_real_i * stabilization_factor_clamped
                stabilized_g_real_list.append(g_real_i)
            if stabilized_g_real_list: g_real = torch.stack(stabilized_g_real_list)
            g_batch = g_real.cfloat() + 1j * g_imag.cfloat()
        except Exception as e: logger.error(f"compute_metric_batch error: {e}", exc_info=True); return torch.eye(hidden_dim, device=DEVICE, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
        return g_batch if is_safe(g_batch) else torch.eye(hidden_dim, device=DEVICE, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)

    def compute_stability_batch(self, g_batch: torch.Tensor) -> torch.Tensor:
        stability_scores = []
        for g in g_batch:
             score = torch.tensor(0.0, device=DEVICE)
             try:
                 g_real = g.real.float()
                 if g_real.numel() > 0 and g_real.ndim == 2 and g_real.shape[0] == g_real.shape[1] and torch.isfinite(g_real).all():
                      eigvals = torch.linalg.eigvals(g_real).abs()
                      if eigvals.numel() > 0 and torch.isfinite(eigvals).all(): score = eigvals.max().clamp(max=20.0)
             except Exception as e: logger.debug(f"Stability calc error: {e}")
             stability_scores.append(score)
        return torch.stack(stability_scores)

    def compressor_batch(self, g_batch: torch.Tensor) -> torch.Tensor:
        try:
            if g_batch.numel() == 0: return torch.zeros(g_batch.shape[0], device=DEVICE)
            traces = torch.stack([torch.trace(g.real.float()) / max(1, g.shape[1]) for g in g_batch])
            return traces.nan_to_num(0.0)
        except Exception as e: logger.warning(f"Compressor batch error: {e}"); return torch.zeros(g_batch.shape[0], device=DEVICE)

    def telezentrik_batch(self, g_batch: torch.Tensor, stability_batch: torch.Tensor) -> torch.Tensor:
        try:
            if g_batch.numel() == 0: return torch.zeros(g_batch.shape[0], device=DEVICE)
            coherence = torch.stack([torch.trace(g.real.float()) / max(1, g.shape[1]) for g in g_batch])
            telezentrik_score = coherence + stability_batch * 0.1
            return telezentrik_score.nan_to_num(0.0)
        except Exception as e: logger.warning(f"Telezentrik batch error: {e}"); return torch.zeros(g_batch.shape[0], device=DEVICE)

    def aondyne_batch_approx(self, current_belief_batch: Optional[torch.Tensor], dt=None) -> torch.Tensor:
        effective_dt = dt if dt is not None else Config.DT
        batch_size = current_belief_batch.shape[0] if current_belief_batch is not None else 0
        if batch_size == 0 or self.prev_belief_embedding is None or current_belief_batch.shape[1:] != self.prev_belief_embedding.shape:
            return torch.zeros(batch_size, device=DEVICE)
        try:
            prev_repeated = self.prev_belief_embedding.float().unsqueeze(0).expand(batch_size, -1)
            delta_belief = current_belief_batch.float() - prev_repeated
            safe_dt = max(effective_dt, 1e-6)
            aondyne_norms = torch.norm(delta_belief / safe_dt, dim=1).clamp(max=100.0)
            return aondyne_norms
        except Exception as e: logger.warning(f"Aondyne batch approx error: {e}"); return torch.zeros(batch_size, device=DEVICE)

    def compute_reflexivity_batch_approx(self, current_belief_batch: torch.Tensor) -> torch.Tensor:
        batch_size = current_belief_batch.shape[0]
        if self.prev_belief_embedding is None or current_belief_batch.numel() == 0: return torch.zeros(batch_size, device=DEVICE)
        try:
            mean_current = current_belief_batch.mean(dim=0).float()
            prev_belief = self.prev_belief_embedding.float()
            if not is_safe(mean_current) or not is_safe(prev_belief) or mean_current.norm() < 1e-8 or prev_belief.norm() < 1e-8: return torch.zeros(batch_size, device=DEVICE)
            rho_score = F.cosine_similarity(mean_current, prev_belief, dim=0)
            return rho_score.clamp(min=0.0, max=1.0).repeat(batch_size)
        except Exception as e: logger.warning(f"Reflexivity batch error: {e}"); return torch.zeros(batch_size, device=DEVICE)

    def compute_integration_batch(self, belief_embedding_batch: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
        batch_size = belief_embedding_batch.shape[0] if belief_embedding_batch is not None else 0
        integration_scores = torch.zeros(batch_size, 1, device=DEVICE)
        if batch_size == 0: return integration_scores
        try:
            batch_corr_score = torch.tensor(0.0, device=DEVICE)
            if batch_size > 1 and belief_embedding_batch.shape[1] > 1:
                embeddings_float = belief_embedding_batch.float(); mean = torch.mean(embeddings_float, dim=0, keepdim=True); centered_data = embeddings_float - mean
                std_devs = torch.std(embeddings_float, dim=0, unbiased=True).clamp(min=1e-6)
                if torch.isfinite(centered_data).all() and torch.isfinite(std_devs).all():
                    cov_matrix = torch.matmul(centered_data.T, centered_data) / (batch_size - 1)
                    if torch.isfinite(cov_matrix).all():
                         outer_std = torch.outer(std_devs, std_devs)
                         if torch.isfinite(outer_std).all():
                              corr_matrix = cov_matrix / outer_std.clamp(min=1e-9)
                              mask = ~torch.eye(corr_matrix.shape[0], dtype=torch.bool, device=DEVICE)
                              if mask.any() and corr_matrix[mask].numel() > 0: batch_corr_score = corr_matrix[mask].abs().mean().nan_to_num(0.0)
            integration_scores.fill_(batch_corr_score)
            if g_batch is not None and g_batch.ndim == 3 and g_batch.shape[0] == batch_size and is_safe(g_batch):
                traces = torch.stack([torch.trace(g.real.float()) / max(1, g.shape[1]) for g in g_batch]).nan_to_num(0.0)
                integration_scores += traces.unsqueeze(1) * 0.1
            elif g_batch is not None: logger.warning(f"g_batch shape {g_batch.shape} incompatible in compute_integration.")
        except Exception as e: logger.warning(f"Integration batch error: {e}")
        return integration_scores.clamp(min=0.0, max=2.0)

    def compute_dynamic_threshold_batch(self, stability_batch: torch.Tensor, belief_change_norm_batch: torch.Tensor) -> torch.Tensor:
        batch_size = stability_batch.shape[0]; entropy_proxy = self.entropy_proxy; entropy_batch = entropy_proxy.repeat(batch_size)
        clamped_stability = stability_batch.clamp(max=5.0); clamped_dynamics = belief_change_norm_batch.clamp(max=1.0)
        tau_0=Config.RL.TAU_0_BASE; stab_term=Config.RL.BETA_TELEZENTRIK*clamped_stability; dyn_term=Config.RL.DYNAMICS_THRESHOLD_WEIGHT*clamped_dynamics; ent_term=Config.RL.ALPHA_TAU_DYNAMICS*entropy_batch
        tau_t_batch = tau_0 - stab_term - dyn_term + ent_term
        return tau_t_batch.clamp(min=0.1).unsqueeze(1)

    def compute_lyapunov_proxy_batch(self, aondyne_batch: torch.Tensor) -> torch.Tensor:
        return aondyne_batch.unsqueeze(1)

    def compute_geometry_batch(self, state_history_batch: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Gamma (connection) and R (curvature) proxies using state history.
        Currently uses the agent's single snapshot history as an approximation for the batch.
        """
        # Determine batch size safely
        batch_size = 1
        if state_history_batch is not None and state_history_batch.ndim > 1:
             # Assuming history shape [B, HistLen, Dim] or [HistLen, Dim] if passed from single forward
             if state_history_batch.ndim == 3:
                 batch_size = state_history_batch.shape[0]
             elif state_history_batch.ndim == 2: # Might be single history passed
                 batch_size = 1 # Assume batch size 1 if only 2 dims provided
             # else: keep batch_size = 1

        # Initialize proxies with correct batch dimension
        gamma_proxy = torch.zeros(batch_size, self.state_dim, device=DEVICE)
        R_proxy = torch.zeros(batch_size, self.state_dim, device=DEVICE)

        # Use the agent's internal snapshot history for calculation (approximation)
        single_history = self.state_history
        if single_history.shape[0] >= 3: # Need at least 3 points
            try:
                hist_f = single_history.float()
                delta1 = hist_f[1:] - hist_f[:-1] # Shape [HistLen-1, Dim]
                delta2 = delta1[1:] - delta1[:-1] # Shape [HistLen-2, Dim]

                # Use last values if available
                last_gamma = delta1[-1] if delta1.shape[0] > 0 else torch.zeros(self.state_dim, device=DEVICE)
                last_R = delta2[-1] if delta2.shape[0] > 0 else torch.zeros(self.state_dim, device=DEVICE)

                # Expand the single result to match batch size
                gamma_proxy = last_gamma.unsqueeze(0).expand(batch_size, -1)
                R_proxy = last_R.unsqueeze(0).expand(batch_size, -1)

            except Exception as e:
                logger.warning(f"Geometry proxy calculation error: {e}")
                # Return zeros if calculation fails
                gamma_proxy = torch.zeros(batch_size, self.state_dim, device=DEVICE)
                R_proxy = torch.zeros(batch_size, self.state_dim, device=DEVICE)
        # else: calculation not possible, return zeros (already initialized)

        return gamma_proxy, R_proxy
    def qualia_map(self, transcendent_embedding: torch.Tensor, emotions: torch.Tensor) -> torch.Tensor:
        if transcendent_embedding is None or transcendent_embedding.numel() == 0: return torch.zeros(self.qualia_dim, device=DEVICE)
        if transcendent_embedding.ndim > 1: transcendent_embedding = transcendent_embedding[0]
        return torch.sigmoid(self.qualia_output_head(transcendent_embedding))

    # === Batched Forward Pass ===
    def forward_batch(self, states_batch: torch.Tensor, rewards_batch: torch.Tensor, state_history_batch: Optional[torch.Tensor]=None, use_target_net: bool = False) -> Optional[AgentForwardReturnType]:
        """ Performs the full forward pass for a batch using PyG. Returns None on critical error. """
        try:
            batch_size = states_batch.shape[0]
            states_batch=states_batch.float().to(DEVICE); rewards_batch=rewards_batch.float().to(DEVICE)
            history_snapshot_batch = state_history_batch if state_history_batch is not None else self.state_history.unsqueeze(0).expand(batch_size, -1, -1)

            encoder = self.target_encoder if use_target_net else self.encoder
            gnn_module = self.target_gnn_layers_module if use_target_net else self.gnn_layers_module
            reflect_layer = self.target_self_reflect_layer if use_target_net else self.self_reflect_layer
            value_head = self.target_value_head if use_target_net else self.value_head
            emo_module = self.emotional_module; qualia_head = self.qualia_output_head; feedback_proj = self.feedback

            # 1. Emotional Update (only needed for online)
            prev_emo_batch = self.prev_emotions.unsqueeze(0).expand(batch_size, -1)
            emotions_batch = emo_module(states_batch[:, :self.emotion_dim], rewards_batch, prev_emo_batch) if not use_target_net else prev_emo_batch.detach()

            # 2. Encode & Build PyG Batch
            encoded_state_batch = F.relu(encoder(states_batch))
            graph_list = [self.build_graph(encoded_state_batch[i]) for i in range(batch_size)]
            graph_pyg_batch = Batch.from_data_list(graph_list).to(DEVICE)

            # 3. Run GNN
            belief_embedding_batch = self.run_gnn(graph_pyg_batch, use_target=use_target_net)

            # 4. Reflexion
            transcendent_embedding_batch = F.relu(reflect_layer(belief_embedding_batch))

            # 5. Value Prediction
            value_pred_batch = value_head(transcendent_embedding_batch) # V(s) or V(s')

            # --- Metrics calculated only for ONLINE pass ---
            if not use_target_net:
                aondyne_batch = self.aondyne_batch_approx(belief_embedding_batch)
                g_batch = self.compute_metric_batch(transcendent_embedding_batch)
                zeta_batch = self.compressor_batch(g_batch); stability_batch = self.compute_stability_batch(g_batch)
                telezentrik_t_batch = self.telezentrik_batch(g_batch, stability_batch)
                lyapunov_max_proxy_batch = self.compute_lyapunov_proxy_batch(aondyne_batch)
                gamma_batch, R_batch = self.compute_geometry_batch(history_snapshot_batch)
                if belief_embedding_batch.numel() > 0: belief_softmax = F.softmax(belief_embedding_batch.float(), dim=-1); self.entropy_proxy = -torch.sum(belief_softmax * torch.log(belief_softmax + 1e-9), dim=-1).clamp(min=0.0).mean()
                else: self.entropy_proxy = torch.tensor(0.0, device=DEVICE)
                integration_I_batch = self.compute_integration_batch(belief_embedding_batch, g_batch)
                rho_score_batch = self.compute_reflexivity_batch_approx(belief_embedding_batch)
                tau_t_batch = self.compute_dynamic_threshold_batch(stability_batch, aondyne_batch)
                att_score_batch = torch.vmap(lambda g: g.real.diagonal().mean())(g_batch).nan_to_num(0.0)
                g_norm_batch = torch.vmap(lambda g: g.real.norm())(g_batch).nan_to_num(1e-9) + 1e-9
                self_consistency_batch = torch.vmap(lambda g, n: torch.trace(g.real @ g.real.T) / (n * n))(g_batch, g_norm_batch).nan_to_num(0.0)
                qualia_features_batch = torch.sigmoid(qualia_head(transcendent_embedding_batch))
                full_state_new_batch = torch.cat([emotions_batch, qualia_features_batch], dim=1)
                feedback_batch = self.lattice.discretize(feedback_proj(transcendent_embedding_batch))
                self.prev_belief_embedding = belief_embedding_batch[-1].detach().clone()
                self.selector = 0.98 * self.selector + 0.02 * stability_batch.mean().clamp(max=10.0)

                return AgentForwardReturnType(
                    emotions=emotions_batch.detach(), belief_embedding=belief_embedding_batch.detach(),
                    integration_I=integration_I_batch.detach(), rho_score=rho_score_batch.unsqueeze(1).detach(),
                    zeta=zeta_batch.unsqueeze(1).detach(), feedback=feedback_batch.detach(), stability=stability_batch.unsqueeze(1).detach(),
                    full_state=full_state_new_batch.detach(), gamma=gamma_batch.detach(), R=R_batch.detach(),
                    att_score=att_score_batch.unsqueeze(1).detach(), self_consistency=self_consistency_batch.unsqueeze(1).detach(),
                    tau_t=tau_t_batch.detach(), metric_g=g_batch.detach(), value_pred=value_pred_batch,
                    lyapunov_max_proxy=lyapunov_max_proxy_batch.detach() )
            else: # Target net only returns V(s') prediction
                  return AgentForwardReturnType( *( (value_pred_batch,) + (None,) * (len(AgentForwardReturnType._fields) - 1) ) ) # Use None for all other fields

        except Exception as e:
             logger.error(f"Error in forward_batch (use_target={use_target_net}): {e}", exc_info=True)
             return None # Indicate failure

    # === Learn method (Handles Batch Processing) ===
    @torch.enable_grad()
    def learn(self, batch_data: Dict[str, torch.Tensor], indices: torch.Tensor, weights: torch.Tensor) -> float:
        """ Performs learning update on a batch, calculates combined loss, returns avg loss scalar. """
        self.train(); self.set_target_networks_eval()
        loss_val = 0.0
        try:
            states = batch_data['states']; rewards = batch_data['rewards'];
            next_states = batch_data['next_states']; dones = batch_data['dones']
            current_batch_size = states.shape[0]
            if current_batch_size == 0: return 0.0

            # --- Online Network Pass ---
            online_out = self.forward_batch(states, rewards, None, use_target_net=False)
            if online_out is None or online_out.value_pred is None: logger.error("Online forward failed."); return 0.0
            q_values_s = online_out.value_pred

            # --- Target Network Pass ---
            with torch.no_grad():
                 target_out = self.forward_batch(next_states, torch.zeros_like(rewards), None, use_target_net=True)
                 if target_out is None or target_out.value_pred is None: logger.error("Target forward failed."); return 0.0
                 q_values_sp = target_out.value_pred
                 q_targets = rewards + Config.RL.GAMMA * q_values_sp * (~dones)

            # --- Value Loss (TD Error) ---
            td_error = q_targets - q_values_s
            if not is_safe(td_error): td_error = torch.nan_to_num(td_error).clamp(-10, 10)
            value_loss = F.smooth_l1_loss(q_values_s, q_targets.detach(), reduction='none', beta=1.0)
            weighted_value_loss = (weights * value_loss).mean() * Config.RL.VALUE_LOSS_WEIGHT

            # --- Syntrometric/RIH/HET Loss Components ---
            # Safely access metrics, providing defaults if None
            integration_I = online_out.integration_I if online_out.integration_I is not None else torch.zeros_like(rewards)
            rho_score = online_out.rho_score if online_out.rho_score is not None else torch.zeros_like(rewards)
            tau_t = online_out.tau_t if online_out.tau_t is not None else torch.ones_like(rewards) * Config.RL.TAU_0_BASE
            stability = online_out.stability if online_out.stability is not None else torch.zeros_like(rewards)
            lyapunov_proxy = online_out.lyapunov_max_proxy if online_out.lyapunov_max_proxy is not None else torch.zeros_like(rewards)
            zeta = online_out.zeta if online_out.zeta is not None else torch.zeros_like(rewards)
            R_norm = torch.norm(online_out.R, dim=1, keepdim=True) if online_out.R is not None else torch.zeros_like(rewards)
            self_consistency = online_out.self_consistency if online_out.self_consistency is not None else torch.zeros_like(rewards)
            telezentrik_t = self.telezentrik_batch(online_out.metric_g, stability) if online_out.metric_g is not None else torch.zeros_like(rewards)
            if telezentrik_t.ndim == 1: telezentrik_t = telezentrik_t.unsqueeze(1)

            rih_integration_penalty = F.relu(tau_t - integration_I)
            rih_reflexivity_penalty = F.relu(Config.RL.RHO_SIMILARITY_THRESHOLD - rho_score)
            stability_penalty = F.relu(stability - self.selector.detach())**2
            dynamical_stability_penalty = F.relu(lyapunov_proxy)
            geometric_coherence_penalty = torch.tensor(0.0, device=DEVICE) # Placeholder
            telezentrik_reward_term = -telezentrik_t
            consistency_reward_term = -self_consistency
            complexity_penalty = torch.abs(zeta)
            curvature_penalty = R_norm

            syntrometric_loss = ( Config.RL.INTEGRATION_WEIGHT * rih_integration_penalty + Config.RL.REFLEXIVITY_WEIGHT * rih_reflexivity_penalty + Config.RL.STABILITY_WEIGHT * stability_penalty + Config.RL.DYNAMICAL_STABILITY_WEIGHT * dynamical_stability_penalty + Config.RL.GEOMETRIC_COHERENCE_WEIGHT * geometric_coherence_penalty + Config.RL.COMPLEXITY_WEIGHT * complexity_penalty + Config.RL.CURVATURE_WEIGHT * curvature_penalty + Config.RL.TELEZENTRIK_WEIGHT * telezentrik_reward_term + Config.RL.CONSISTENCY_WEIGHT * consistency_reward_term ).mean()

            # --- Total Loss ---
            total_loss = weighted_value_loss + syntrometric_loss

            # --- Optimization ---
            self.optimizer.zero_grad()
            loss_val = 0.0
            if is_safe(total_loss) and total_loss.requires_grad:
                 total_loss.backward()
                 trainable_params = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
                 if trainable_params: torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=Config.RL.GRADIENT_CLIP_AGENT)
                 self.optimizer.step()
                 loss_val = total_loss.item()
            else: logger.warning(f"Skipping opt step. Loss unsafe or no grad: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss}")

            # --- PER Priority Update ---
            priorities = td_error.abs().detach().squeeze() + 1e-5
            self.memory.update_priorities(indices, priorities)

            # --- Target Network Soft Update ---
            if self.step_count % Config.RL.TARGET_NETWORK_UPDATE_FREQ == 0: self.soft_update_target_networks()

            # --- Add experiences to memory AFTER update ---
            beliefs_for_mem = online_out.belief_embedding
            if beliefs_for_mem is not None and beliefs_for_mem.shape[0] == current_batch_size:
                for i in range(current_batch_size):
                     exp = Experience( states[i].detach().cpu(), beliefs_for_mem[i].detach().cpu(), rewards[i].item(), next_states[i].detach().cpu(), dones[i].item(), td_error[i].item() )
                     self.memory.add(exp)
            else: logger.warning("Skipping memory add due to invalid belief embeddings.")

            self.step_count += 1; self.beta = min(1.0, self.beta + self.beta_increment)

        except Exception as learn_err: logger.error(f"Error during agent learn step: {learn_err}", exc_info=True); loss_val = 0.0
        return loss_val

    # === Single Instance Forward (Wrapper) ===
    def forward_single(self, state: torch.Tensor, reward: float, state_history: torch.Tensor) -> Optional[AgentForwardReturnType]:
        """ Runs forward pass for a single state instance. Returns None on error. """
        self.eval()
        result = None
        try:
            state_batch = state.unsqueeze(0); reward_batch = torch.tensor([[reward]], device=DEVICE)
            history_batch = state_history.unsqueeze(0) if state_history.ndim == 2 else state_history
            with torch.no_grad(): result = self.forward_batch(state_batch, reward_batch, history_batch, use_target_net=False)
        except Exception as e: logger.error(f"forward_single error: {e}", exc_info=True)
        finally: self.train()
        return result

    # === Generate Response (Contextual) ===
    def generate_response(self, context: str, att_score: float) -> str:
        """ Generates text response using TransformerGPT, prepending internal state. """
        if not self.gpt: return "[GPT Error: Not Initialized]"
        emo_summary = "neutral"; clamped_att = max(0, min(1, att_score))
        if hasattr(self, 'prev_emotions') and self.prev_emotions is not None:
            try: emo_cpu = self.prev_emotions.cpu().numpy(); emo_summary = f"feeling({emo_cpu.round(2)})"
            except Exception: pass
        state_context = f"[Internal: {emo_summary}, Att={clamped_att:.2f}] "; full_prompt = state_context + "\n" + context
        temp = Config.NLP.GPT_TEMPERATURE * (1.0 + (1.0 - clamped_att)*0.2)
        return self.gpt.generate(full_prompt, temperature=temp, top_p=Config.NLP.GPT_TOP_P)

    # === Save/Load State (Includes ALL target nets) ===
    def save_state(self, agent_path=AGENT_SAVE_PATH, gpt_path=GPT_SAVE_PATH, optimizer_path=OPTIMIZER_SAVE_PATH, target_path_suffix=None): # Removed suffix default arg
        logger.info(f"Saving agent state to {agent_path}...")
        try:
            os.makedirs(os.path.dirname(agent_path), exist_ok=True)
            agent_state = {
                'encoder_state_dict': self.encoder.state_dict(),
                'gnn_layers_module_state_dict': self.gnn_layers_module.state_dict(),
                'emotional_module_state_dict': self.emotional_module.state_dict(),
                'self_reflect_layer_state_dict': self.self_reflect_layer.state_dict(),
                'feedback_state_dict': self.feedback.state_dict(),
                'qualia_output_head_state_dict': self.qualia_output_head.state_dict(),
                'value_head_state_dict': self.value_head.state_dict(),
                # Save target states directly in the main file now
                'target_encoder_state_dict': self.target_encoder.state_dict(),
                'target_gnn_layers_module_state_dict': self.target_gnn_layers_module.state_dict(),
                'target_self_reflect_layer_state_dict': self.target_self_reflect_layer.state_dict(),
                'target_value_head_state_dict': self.target_value_head.state_dict(),
                # Metadata
                'prev_emotions': self.prev_emotions, 'prev_belief_embedding': self.prev_belief_embedding,
                'selector': self.selector, 'step_count': self.step_count, 'beta': self.beta,
                'state_dim': self.state_dim, 'hidden_dim': self.hidden_dim, 'gnn_hidden_dim': self.gnn_hidden_dim,
            }
            torch.save(agent_state, agent_path)
            logger.debug(f"Core agent and target states saved to {agent_path}")
            self.gpt.save_model(gpt_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            logger.info("Agent, GPT, and optimizer states saved successfully.")
        except Exception as e: logger.error(f"Error saving agent state: {e}", exc_info=True)

    def load_state(self, agent_path=AGENT_SAVE_PATH, gpt_path=GPT_SAVE_PATH, optimizer_path=OPTIMIZER_SAVE_PATH, target_path_suffix=None) -> bool: # target_path_suffix ignored
        logger.info(f"Loading agent state from {agent_path}...")
        if not os.path.exists(agent_path): logger.error(f"Agent state file not found: {agent_path}."); return False
        try:
            agent_state = torch.load(agent_path, map_location=DEVICE)
            if agent_state.get('state_dim')!=self.state_dim or agent_state.get('hidden_dim')!=self.hidden_dim or agent_state.get('gnn_hidden_dim')!=self.gnn_hidden_dim:
                 logger.critical("CRITICAL LOAD ERROR: Dimension mismatch."); return False
            # Load Online Networks
            self.encoder.load_state_dict(agent_state['encoder_state_dict']); self.gnn_layers_module.load_state_dict(agent_state['gnn_layers_module_state_dict']); self.emotional_module.load_state_dict(agent_state['emotional_module_state_dict']); self.self_reflect_layer.load_state_dict(agent_state['self_reflect_layer_state_dict']); self.feedback.load_state_dict(agent_state['feedback_state_dict']); self.qualia_output_head.load_state_dict(agent_state['qualia_output_head_state_dict']); self.value_head.load_state_dict(agent_state['value_head_state_dict'])
            # Load Target Networks from the same file
            self.target_encoder.load_state_dict(agent_state['target_encoder_state_dict']); self.target_gnn_layers_module.load_state_dict(agent_state['target_gnn_layers_module_state_dict']); self.target_self_reflect_layer.load_state_dict(agent_state['target_self_reflect_layer_state_dict']); self.target_value_head.load_state_dict(agent_state['target_value_head_state_dict'])
            self.set_target_networks_eval()
            # Load Metadata
            self.prev_emotions = agent_state.get('prev_emotions', torch.zeros_like(self.prev_emotions)).to(DEVICE)
            self.prev_belief_embedding = agent_state.get('prev_belief_embedding');
            if self.prev_belief_embedding is not None: self.prev_belief_embedding = self.prev_belief_embedding.to(DEVICE)
            self.selector = agent_state.get('selector', torch.tensor(1.0, device=DEVICE)).to(DEVICE)
            self.step_count = agent_state.get('step_count', 0); self.beta = agent_state.get('beta', Config.RL.PER_BETA_START)
            logger.info("Agent & Target networks and metadata loaded.")
            # Load GPT & Optimizer
            if not self.gpt.load_model(gpt_path): logger.warning(f"GPT load failed {gpt_path}. Using base."); self.gpt = TransformerGPT()
            if os.path.exists(optimizer_path):
                 self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
                 online_params = list(self.encoder.parameters()) + list(self.gnn_layers_module.parameters()) + \
                                list(self.emotional_module.parameters()) + list(self.self_reflect_layer.parameters()) + \
                                list(self.feedback.parameters()) + list(self.qualia_output_head.parameters()) + \
                                list(self.value_head.parameters())
                 self.optimizer.param_groups.clear(); self.optimizer.add_param_group({'params': [p for p in online_params if p.requires_grad]})
                 for group in self.optimizer.param_groups: group['lr'] = Config.RL.LR
                 logger.info("Optimizer state loaded.")
            else: logger.warning(f"Optimizer state not found: {optimizer_path}.")
            self.eval(); # Set online nets to eval initially
            return True
        except Exception as e: logger.error(f"Load state error: {e}", exc_info=True); return False

# --- END OF FILE agent.py ---

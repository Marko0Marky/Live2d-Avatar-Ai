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

# FIX: Import config essentials (logger, DEVICE) at the TOP
from config import MasterConfig as Config, DEVICE, logger
# --- End FIX ---

# --- PyTorch Geometric Imports ---
try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data, Batch
    from torch_scatter import scatter_mean, scatter_add, scatter_max
    PYG_AVAILABLE = True
    logger.info("PyTorch Geometric and torch_scatter found.")
except ImportError:
    # Logger is now defined here too
    logger.critical("PyTorch Geometric (PyG) or torch_scatter NOT FOUND. Agent cannot function. Please install using official PyG guidelines matching your PyTorch/CUDA version.")
    PYG_AVAILABLE = False
    sys.exit(1) # Critical dependency

# Import other config/utils/modules AFTER checking PyG
from config import AGENT_SAVE_PATH, GPT_SAVE_PATH, OPTIMIZER_SAVE_PATH, TARGET_NET_SAVE_SUFFIX
from utils import MetronicLattice, MetaCognitiveMemory, is_safe, Experience
from ai_modules import EmotionalModule, TransformerGPT

# Define the comprehensive return type for the forward pass
AgentForwardReturnType = namedtuple('AgentForwardReturnType', [
    'emotions', 'belief_embedding', 'integration_I', 'rho_score', 'zeta',
    'feedback', 'stability', 'full_state', 'gamma', 'R', 'att_score',
    'self_consistency', 'tau_t', 'metric_g', 'value_pred', 'lyapunov_max_proxy'
])

class ConsciousAgent(nn.Module):
    def __init__(self, state_dim: int = Config.Agent.STATE_DIM, hidden_dim: int = Config.Agent.HIDDEN_DIM):
        super().__init__()
        if state_dim != 12: logger.critical(f"Agent requires STATE_DIM=12."); sys.exit(1)
        if not PYG_AVAILABLE: logger.critical("PyG not available."); sys.exit(1)

        self.state_dim = state_dim; self.hidden_dim = hidden_dim;
        self.emotion_dim = Config.Agent.EMOTION_DIM; self.qualia_dim = self.state_dim - self.emotion_dim
        self.gnn_hidden_dim = Config.Agent.GNN.GNN_HIDDEN_DIM

        # --- Online Networks ---
        self.lattice = MetronicLattice(dim=self.state_dim, tau=Config.Agent.TAU)
        self.encoder = nn.Linear(self.state_dim, self.gnn_hidden_dim).to(DEVICE)
        self.emotional_module = EmotionalModule(input_dim=self.emotion_dim + 1).to(DEVICE)
        self.gnn_layers_module = self._build_gnn_module().to(DEVICE)
        self.self_reflect_layer = nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim).to(DEVICE)
        self.qualia_output_head = nn.Linear(self.gnn_hidden_dim, self.qualia_dim).to(DEVICE) # Maps belief to qualia dims
        self.feedback = nn.Linear(self.gnn_hidden_dim, self.state_dim).to(DEVICE) # Feedback projection
        self.value_head = nn.Linear(self.gnn_hidden_dim, 1).to(DEVICE) # State value predictor

        # --- Target Networks ---
        self.target_encoder = nn.Linear(self.state_dim, self.gnn_hidden_dim).to(DEVICE)
        self.target_gnn_layers_module = self._build_gnn_module().to(DEVICE)
        self.target_self_reflect_layer = nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim).to(DEVICE)
        self.target_value_head = nn.Linear(self.gnn_hidden_dim, 1).to(DEVICE)

        self.update_target_network() # Initialize targets with online weights
        self.set_target_networks_eval() # Target nets only used for inference
        self.soft_update_tau = Config.RL.TARGET_NETWORK_SOFT_UPDATE_TAU

        try: self.gpt = TransformerGPT()
        except Exception as e: logger.critical(f"GPT Init Error: {e}."); sys.exit(1)
        self.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE)

        # Optimizer includes all online network parameters
        online_params = list(self.encoder.parameters()) + list(self.gnn_layers_module.parameters()) + \
                        list(self.emotional_module.parameters()) + list(self.self_reflect_layer.parameters()) + \
                        list(self.feedback.parameters()) + list(self.qualia_output_head.parameters()) + \
                        list(self.value_head.parameters())
        self.optimizer = optim.AdamW([p for p in online_params if p.requires_grad], lr=Config.RL.LR, weight_decay=0.001) # Use AdamW

        # State Tracking
        self.state_history_deque: Deque[torch.Tensor] = deque(maxlen=Config.Agent.HISTORY_SIZE)
        for _ in range(Config.Agent.HISTORY_SIZE): self.state_history_deque.append(torch.zeros(self.state_dim, device=DEVICE))
        self.prev_emotions = torch.zeros(self.emotion_dim, device=DEVICE)
        self.prev_belief_embedding: Optional[torch.Tensor] = None # Store last single belief embedding for reflexivity approx
        self.selector = torch.tensor(1.0, device=DEVICE, dtype=torch.float32) # Dynamic stability selector
        self.step_count = 0; self.beta = Config.RL.PER_BETA_START
        beta_frames = Config.RL.PER_BETA_FRAMES; self.beta_increment = (1.0 - self.beta) / max(1, beta_frames)
        self.entropy_proxy = torch.tensor(0.0, device=DEVICE) # Store average entropy

        logger.info(f"ConsciousAgent initialized with PyG {Config.Agent.GNN.GNN_TYPE} layers and Target Networks.")

    def _create_gnn_layer(self, in_channels, out_channels):
        """Creates a PyG GNN layer based on config."""
        gnn_type=Config.Agent.GNN.GNN_TYPE.upper(); heads=Config.Agent.GNN.GAT_HEADS
        dropout_rate = 0.1 # Example dropout
        if gnn_type=='GCN': layer = pyg_nn.GCNConv(in_channels, out_channels)
        elif gnn_type=='GAT': layer = pyg_nn.GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout_rate) # Note: Output dim is out_channels if concat=False
        elif gnn_type=='GRAPHSAGE': layer = pyg_nn.SAGEConv(in_channels, out_channels)
        elif gnn_type=='GIN': mlp=nn.Sequential(nn.Linear(in_channels, out_channels*2), nn.ReLU(), nn.Linear(out_channels*2, out_channels)); layer = pyg_nn.GINConv(mlp)
        else: logger.warning(f"Unknown GNN_TYPE '{gnn_type}'. Using GCNConv."); layer = pyg_nn.GCNConv(in_channels, out_channels)
        return layer.to(DEVICE) # Ensure layer is on device

    def _build_gnn_module(self) -> nn.ModuleList:
        """Builds the stack of GNN layers."""
        gnn_layers = nn.ModuleList()
        current_channels = self.gnn_hidden_dim
        for i in range(Config.Agent.GNN.GNN_LAYERS):
            layer = self._create_gnn_layer(current_channels, self.gnn_hidden_dim)
            norm = nn.LayerNorm(self.gnn_hidden_dim).to(DEVICE)
            # Store layers in a ModuleDict for clarity if needed, or just append directly
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
             # Handle ModuleList structure correctly
             if isinstance(base_net, nn.ModuleList) and isinstance(target_net, nn.ModuleList):
                 for target_module_dict, base_module_dict in zip(target_net, base_net):
                     # Iterate through the layers within the ModuleDict
                     for key in base_module_dict:
                         if isinstance(base_module_dict[key], nn.Module): # Ensure it's a layer
                             for target_p, base_p in zip(target_module_dict[key].parameters(), base_module_dict[key].parameters()):
                                  target_p.data.mul_(1.0 - tau); target_p.data.add_(tau * base_p.data)
             elif isinstance(base_net, nn.Module) and isinstance(target_net, nn.Module):
                 for target_p, base_p in zip(target_net.parameters(), base_net.parameters()):
                      target_p.data.mul_(1.0 - tau); target_p.data.add_(tau * base_p.data)

    def soft_update_target_networks(self, tau=None):
        """Soft updates all target networks."""
        update_tau = tau if tau is not None else self.soft_update_tau
        self._polyak_update(self.encoder, self.target_encoder, update_tau)
        self._polyak_update(self.gnn_layers_module, self.target_gnn_layers_module, update_tau)
        self._polyak_update(self.self_reflect_layer, self.target_self_reflect_layer, update_tau)
        self._polyak_update(self.value_head, self.target_value_head, update_tau)

    @property
    def state_history(self) -> torch.Tensor:
        """Returns the 12D state history as a stacked tensor."""
        if not self.state_history_deque: return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)
        valid = all(isinstance(t, torch.Tensor) and t.shape == (self.state_dim,) for t in self.state_history_deque)
        if not valid or len(self.state_history_deque) < Config.Agent.HISTORY_SIZE:
             logger.warning(f"Agent history invalid/incomplete. Reinitializing.")
             self.state_history_deque.clear(); initial_state = torch.zeros(self.state_dim, device=DEVICE)
             for _ in range(Config.Agent.HISTORY_SIZE): self.state_history_deque.append(initial_state.clone())
        try: return torch.stack(list(self.state_history_deque)).to(DEVICE).float()
        except Exception as e: logger.error(f"Error stacking agent history: {e}."); return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)

    # === Helper Methods for Syntrometric Computations ===

    def build_graph(self, state_embedding: torch.Tensor) -> Data:
        """ Builds a PyG Data object for a single state embedding. """
        num_nodes = 7; node_features = state_embedding.unsqueeze(0).repeat(num_nodes, 1)
        edge_list = [[0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 6], [4, 6]];
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=DEVICE).t().contiguous()
        edge_attr = torch.ones(edge_index.shape[1], Config.Agent.GNN.GAT_HEADS, device=DEVICE) if Config.Agent.GNN.GNN_TYPE == 'GAT' else None # Edge attributes for GAT
        return Data(x=node_features.float(), edge_index=edge_index, edge_attr=edge_attr)

    def run_gnn(self, graph_batch: Batch, use_target=False) -> torch.Tensor:
        """ Processes a PyG Batch object through GNN layers. """
        layers = self.target_gnn_layers_module if use_target else self.gnn_layers_module
        x, edge_index, batch_map = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        edge_attr = getattr(graph_batch, 'edge_attr', None)
        if not is_safe(x): logger.warning("Unsafe input to run_gnn"); x = torch.zeros_like(x)

        for i, layer_block in enumerate(layers):
            x_res = x
            conv = layer_block['conv']; norm = layer_block['norm']
            try:
                 # Pass edge_attr only if it exists and the layer type might use it (e.g., GAT)
                 if edge_attr is not None and Config.Agent.GNN.GNN_TYPE == 'GAT':
                     x = conv(x, edge_index, edge_attr=edge_attr)
                 # Some layers might accept edge_weight instead or as well
                 # elif hasattr(graph_batch, 'edge_weight') and ... :
                 #     x = conv(x, edge_index, edge_weight=graph_batch.edge_weight)
                 else:
                     x = conv(x, edge_index)
                 x = F.relu(x) # Apply ReLU activation
                 x = norm(x)
                 if Config.Agent.GNN.GNN_USE_RESIDUAL and x.shape == x_res.shape: x = x + x_res
                 if not is_safe(x): raise ValueError(f"Unsafe output GNN layer {i}")
            except Exception as e: logger.error(f"GNN layer {i} ({type(conv).__name__}) error: {e}", exc_info=True); return torch.zeros(graph_batch.num_graphs, self.gnn_hidden_dim, device=DEVICE)

        aggregated_embedding = scatter_mean(x, batch_map, dim=0) # Aggregate node features
        return aggregated_embedding if is_safe(aggregated_embedding) else torch.zeros(graph_batch.num_graphs, self.gnn_hidden_dim, device=DEVICE)

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
                        else: max_eig_i = torch.tensor(100.0, device=DEVICE) # Assign high value if eig fails
                    else: max_eig_i = torch.tensor(100.0, device=DEVICE) # Assign high value if matrix not finite
                except torch.linalg.LinAlgError: logger.debug(f"Eig compute failed for sample {i}"); max_eig_i = torch.tensor(100.0, device=DEVICE)
                except Exception as eig_e : logger.warning(f"Metric Eig Err sample {i}: {eig_e}"); max_eig_i = torch.tensor(100.0, device=DEVICE)

                # Compare tensors directly
                if max_eig_i > current_selector:
                    # Keep stabilization_factor as a tensor
                    stabilization_factor = (current_selector / max_eig_i.clamp(min=1e-6))
                    # *** FIX: Clamp the tensor factor ***
                    stabilization_factor_clamped = stabilization_factor.clamp(min=0.01, max=1.0)
                    g_real_i = g_real_i * stabilization_factor_clamped # Multiply tensor by tensor
                stabilized_g_real_list.append(g_real_i)

            if stabilized_g_real_list: g_real = torch.stack(stabilized_g_real_list)
            g_batch = g_real.cfloat() + 1j * g_imag.cfloat() # Use cfloat for complex conversion

        except Exception as e: logger.error(f"compute_metric_batch error: {e}", exc_info=True); return torch.eye(hidden_dim, device=DEVICE, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
        return g_batch if is_safe(g_batch) else torch.eye(hidden_dim, device=DEVICE, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)

    def compute_stability_batch(self, g_batch: torch.Tensor) -> torch.Tensor:
        """ Computes g_ik stability proxy for a batch. """
        stability_scores = []
        for g in g_batch:
             score = torch.tensor(0.0, device=DEVICE)
             try:
                 g_real = g.real.float()
                 if g_real.numel() > 0 and g_real.ndim == 2 and g_real.shape[0] == g_real.shape[1] and torch.isfinite(g_real).all():
                      eigvals = torch.linalg.eigvals(g_real).abs()
                      if eigvals.numel() > 0 and torch.isfinite(eigvals).all(): score = eigvals.max().clamp(max=20.0) # Increased clamp
             except Exception as e: logger.debug(f"Stability calc error: {e}") # Less alarming level
             stability_scores.append(score)
        return torch.stack(stability_scores)

    def compressor_batch(self, g_batch: torch.Tensor) -> torch.Tensor:
        """ Computes zeta proxy (structure condensation) for a batch. """
        try:
            if g_batch.numel() == 0: return torch.zeros(g_batch.shape[0], device=DEVICE)
            # Using torch.vmap might fail if lambda accesses self or non-tensor vars implicitly
            traces = torch.stack([torch.trace(g.real.float()) / max(1, g.shape[1]) for g in g_batch])
            return traces.nan_to_num(0.0)
        except Exception as e: logger.warning(f"Compressor batch error: {e}"); return torch.zeros(g_batch.shape[0], device=DEVICE)

    def telezentrik_batch(self, g_batch: torch.Tensor, stability_batch: torch.Tensor) -> torch.Tensor:
        """ Computes telezentrik proxy for a batch. """
        try:
            if g_batch.numel() == 0: return torch.zeros(g_batch.shape[0], device=DEVICE)
            coherence = torch.stack([torch.trace(g.real.float()) / max(1, g.shape[1]) for g in g_batch])
            telezentrik_score = coherence + stability_batch * 0.1 # Stability contribution
            return telezentrik_score.nan_to_num(0.0)
        except Exception as e: logger.warning(f"Telezentrik batch error: {e}"); return torch.zeros(g_batch.shape[0], device=DEVICE)

    def aondyne_batch_approx(self, current_belief_batch: Optional[torch.Tensor], dt=None) -> torch.Tensor:
        """ Approximates Aondyne (belief change rate) using last known single belief. """
        effective_dt = dt if dt is not None else Config.DT
        batch_size = current_belief_batch.shape[0] if current_belief_batch is not None else 0
        # Check if prev belief exists and shapes are compatible
        if batch_size == 0 or self.prev_belief_embedding is None or current_belief_batch.shape[1:] != self.prev_belief_embedding.shape:
            return torch.zeros(batch_size, device=DEVICE)
        try:
            prev_repeated = self.prev_belief_embedding.float().unsqueeze(0).expand(batch_size, -1) # Use expand
            delta_belief = current_belief_batch.float() - prev_repeated
            safe_dt = max(effective_dt, 1e-6)
            aondyne_norms = torch.norm(delta_belief / safe_dt, dim=1).clamp(max=100.0) # Calculate norm per sample
            return aondyne_norms
        except Exception as e: logger.warning(f"Aondyne batch approx error: {e}"); return torch.zeros(batch_size, device=DEVICE)

    def compute_reflexivity_batch_approx(self, current_belief_batch: torch.Tensor) -> torch.Tensor:
        """ Approximates reflexivity rho using batch mean vs last known single belief. """
        batch_size = current_belief_batch.shape[0]
        if self.prev_belief_embedding is None or current_belief_batch.numel() == 0: return torch.zeros(batch_size, device=DEVICE)
        try:
            mean_current = current_belief_batch.mean(dim=0).float() # Calculate mean across batch
            prev_belief = self.prev_belief_embedding.float()
            if not is_safe(mean_current) or not is_safe(prev_belief): return torch.zeros(batch_size, device=DEVICE)
            if mean_current.norm() < 1e-8 or prev_belief.norm() < 1e-8: return torch.zeros(batch_size, device=DEVICE)
            rho_score = F.cosine_similarity(mean_current, prev_belief, dim=0) # Single score for the batch mean vs prev
            return rho_score.clamp(min=0.0, max=1.0).repeat(batch_size) # Repeat score for batch dim consistency
        except Exception as e: logger.warning(f"Reflexivity batch error: {e}"); return torch.zeros(batch_size, device=DEVICE)

    def compute_integration_batch(self, belief_embedding_batch: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
        """ Computes integration I_S proxy for a batch using correlation + trace. """
        # *** FIX: Define batch_size reliably at the START ***
        # Determine batch size primarily from belief_embedding_batch if possible
        if belief_embedding_batch is not None and belief_embedding_batch.ndim >= 1:
             batch_size = belief_embedding_batch.shape[0]
        # Fallback to g_batch if belief is None or scalar
        elif g_batch is not None and g_batch.ndim > 2: # g_batch shape [B, H, H]
             batch_size = g_batch.shape[0]
        else:
             # Cannot determine batch size, return zeros of size 1
             logger.warning("compute_integration_batch: Could not determine batch size. Returning scalar zero.")
             return torch.zeros(1, 1, device=DEVICE)

        # Handle case where batch_size might be 0 if inputs were empty tensors
        if batch_size == 0:
             logger.warning("compute_integration_batch: Determined batch size is 0. Returning empty tensor.")
             return torch.zeros(0, 1, device=DEVICE)

        # Now initialize scores using the determined batch_size
        integration_scores = torch.zeros(batch_size, 1, device=DEVICE)
        # *** END FIX ***

        try:
            # Correlation component (calculated across batch samples)
            batch_corr_score = torch.tensor(0.0, device=DEVICE)
            # Check validity *after* batch_size is determined
            if batch_size > 1 and belief_embedding_batch is not None and belief_embedding_batch.ndim == 2 and belief_embedding_batch.shape[1] > 1:
                embeddings_float = belief_embedding_batch.float();
                mean = torch.mean(embeddings_float, dim=0, keepdim=True)
                centered_data = embeddings_float - mean
                std_devs = torch.std(embeddings_float, dim=0, unbiased=True).clamp(min=1e-6)
                if torch.isfinite(centered_data).all() and torch.isfinite(std_devs).all():
                    cov_matrix = torch.matmul(centered_data.T, centered_data) / (batch_size - 1)
                    # Check for NaN/Inf in cov_matrix before division
                    if torch.isfinite(cov_matrix).all():
                         outer_std = torch.outer(std_devs, std_devs)
                         # Ensure outer_std is finite and non-zero where needed
                         if torch.isfinite(outer_std).all():
                              corr_matrix = cov_matrix / outer_std.clamp(min=1e-9) # Clamp denominator
                              mask = ~torch.eye(corr_matrix.shape[0], dtype=torch.bool, device=DEVICE)
                              if mask.any() and corr_matrix[mask].numel() > 0:
                                   batch_corr_score = corr_matrix[mask].abs().mean().nan_to_num(0.0)
                         else: logger.warning("Non-finite values in outer product of std_devs.")
                    else: logger.warning("Non-finite values in covariance matrix.")
                else: logger.warning("Non-finite values encountered before correlation calculation.")
            # Fill scores AFTER potential calculation
            integration_scores.fill_(batch_corr_score)

            # Add trace component (calculated per sample)
            if g_batch is not None and g_batch.ndim == 3 and g_batch.shape[0] == batch_size and g_batch.numel() > 0:
                # Ensure g_batch is safe before trace
                if is_safe(g_batch):
                     traces = torch.stack([torch.trace(g.real.float()) / max(1, g.shape[1]) for g in g_batch]).nan_to_num(0.0)
                     integration_scores += traces.unsqueeze(1) * 0.1 # Add weighted trace
                else:
                     logger.warning("Unsafe g_batch tensor in compute_integration_batch trace calculation.")
            elif g_batch is not None:
                 logger.warning(f"g_batch shape ({g_batch.shape}) incompatible with batch_size ({batch_size}) or empty.")

        except Exception as e:
            logger.warning(f"Integration batch error: {e}", exc_info=True) # Log full traceback for warnings too
            # Reset scores to zero on error
            integration_scores = torch.zeros(batch_size, 1, device=DEVICE)

        return integration_scores.clamp(min=0.0, max=2.0) # Clamp potential range

    def compute_dynamic_threshold_batch(self, stability_batch: torch.Tensor, belief_change_norm_batch: torch.Tensor) -> torch.Tensor:
        """ Computes dynamic RIH threshold tau_t for a batch. """
        batch_size = stability_batch.shape[0]; entropy_proxy = self.entropy_proxy; entropy_batch = entropy_proxy.repeat(batch_size)
        clamped_stability = stability_batch.clamp(max=5.0); clamped_dynamics = belief_change_norm_batch.clamp(max=1.0)
        tau_0=Config.RL.TAU_0_BASE; stab_term=Config.RL.BETA_TELEZENTRIK*clamped_stability; dyn_term=Config.RL.DYNAMICS_THRESHOLD_WEIGHT*clamped_dynamics; ent_term=Config.RL.ALPHA_TAU_DYNAMICS*entropy_batch
        tau_t_batch = tau_0 - stab_term - dyn_term + ent_term # Stability/Dynamics increase required integration
        return tau_t_batch.clamp(min=0.1).unsqueeze(1) # Ensure shape [B, 1]

    def compute_lyapunov_proxy_batch(self, aondyne_batch: torch.Tensor) -> torch.Tensor:
        """ Computes Lyapunov stability proxy (positive values indicate instability). """
        return aondyne_batch.unsqueeze(1) # Ensure shape [B, 1]

    def compute_geometry_batch(self, state_history_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes Gamma (connection) and R (curvature) proxies using state history batch. """
        single_history = self.state_history # Use agent's snapshot history as approximation
        gamma_proxy, R_proxy = torch.zeros(self.state_dim, device=DEVICE), torch.zeros(self.state_dim, device=DEVICE)
        if single_history.shape[0] >= 3:
            try:
                hist_f = single_history.float(); delta1 = hist_f[1:] - hist_f[:-1]; delta2 = delta1[1:] - delta1[:-1]
                if delta1.numel() > 0: gamma_proxy = delta1[-1]
                if delta2.numel() > 0: R_proxy = delta2[-1]
            except Exception as e: logger.warning(f"Geom proxy error: {e}")
        # Repeat the single history approximation for the batch size
        batch_size = state_history_batch.shape[0] if state_history_batch is not None and state_history_batch.ndim > 1 else 1
        return gamma_proxy.repeat(batch_size, 1), R_proxy.repeat(batch_size, 1)

    def qualia_map(self, transcendent_embedding: torch.Tensor, emotions: torch.Tensor) -> torch.Tensor:
        """ Maps final embedding to qualia dimensions (R7-12). """
        if transcendent_embedding is None or transcendent_embedding.numel() == 0: return torch.zeros(self.qualia_dim, device=DEVICE)
        # Use dedicated head - assumes input is single embedding [H]
        if transcendent_embedding.ndim > 1: # If somehow batched, take first one
             transcendent_embedding = transcendent_embedding[0]
             emotions = emotions[0] if emotions.ndim > 1 else emotions
        return torch.sigmoid(self.qualia_output_head(transcendent_embedding))

    # === Batched Forward Pass ===
    def forward_batch(self, states_batch: torch.Tensor, rewards_batch: torch.Tensor, state_history_batch: Optional[torch.Tensor]=None, use_target_net: bool = False) -> AgentForwardReturnType:
        """ Performs the full forward pass for a batch using PyG. """
        batch_size = states_batch.shape[0]
        states_batch=states_batch.float().to(DEVICE); rewards_batch=rewards_batch.float().to(DEVICE)
        # Use agent's history snapshot if batch history not provided
        history_snapshot_batch = state_history_batch if state_history_batch is not None else self.state_history.unsqueeze(0).expand(batch_size, -1, -1)

        # Select networks
        encoder = self.target_encoder if use_target_net else self.encoder
        gnn_module = self.target_gnn_layers_module if use_target_net else self.gnn_layers_module
        reflect_layer = self.target_self_reflect_layer if use_target_net else self.self_reflect_layer
        value_head = self.target_value_head if use_target_net else self.value_head
        emo_module = self.emotional_module; qualia_head = self.qualia_output_head; feedback_proj = self.feedback

        # 1. Emotional Update
        prev_emo_batch = self.prev_emotions.unsqueeze(0).expand(batch_size, -1) # Use stored prev emotion for batch
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
        value_pred_batch = value_head(transcendent_embedding_batch)

        # --- Metrics calculated only for ONLINE pass ---
        if not use_target_net:
            aondyne_batch = self.aondyne_batch_approx(belief_embedding_batch) # Approx using prev single belief
            g_batch = self.compute_metric_batch(transcendent_embedding_batch)
            zeta_batch = self.compressor_batch(g_batch); stability_batch = self.compute_stability_batch(g_batch)
            telezentrik_t_batch = self.telezentrik_batch(g_batch, stability_batch)
            lyapunov_max_proxy_batch = self.compute_lyapunov_proxy_batch(aondyne_batch)
            gamma_batch, R_batch = self.compute_geometry_batch(history_snapshot_batch) # Uses snapshot history approx
            # Calculate entropy based on batch belief embedding distribution
            if belief_embedding_batch.numel() > 0: belief_softmax = F.softmax(belief_embedding_batch.float(), dim=-1); self.entropy_proxy = -torch.sum(belief_softmax * torch.log(belief_softmax + 1e-9), dim=-1).clamp(min=0.0).mean()
            else: self.entropy_proxy = torch.tensor(0.0, device=DEVICE)
            integration_I_batch = self.compute_integration_batch(belief_embedding_batch, g_batch)
            rho_score_batch = self.compute_reflexivity_batch_approx(belief_embedding_batch) # Approx using prev single belief
            tau_t_batch = self.compute_dynamic_threshold_batch(stability_batch, aondyne_batch)
            att_score_batch = torch.vmap(lambda g: g.real.diagonal().mean())(g_batch).nan_to_num(0.0)
            g_norm_batch = torch.vmap(lambda g: g.real.norm())(g_batch).nan_to_num(1e-9) + 1e-9
            self_consistency_batch = torch.vmap(lambda g, n: torch.trace(g.real @ g.real.T) / (n * n))(g_batch, g_norm_batch).nan_to_num(0.0)
            # Qualia: Apply head to each item in batch
            qualia_features_batch = torch.sigmoid(qualia_head(transcendent_embedding_batch))
            full_state_new_batch = torch.cat([emotions_batch, qualia_features_batch], dim=1)
            feedback_batch = self.lattice.discretize(feedback_proj(transcendent_embedding_batch))

            # Store last belief from batch for next step's approximation
            self.prev_belief_embedding = belief_embedding_batch[-1].detach().clone()
            # Update dynamic selector EMA using batch average stability
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
            return AgentForwardReturnType( *( (value_pred_batch,) + (None,) * 14 ) ) # value_pred is first, rest are None

    # === Learn method (Handles Batch Processing) ===
    @torch.enable_grad() # Ensure gradients are enabled for learning
    def learn(self, batch_data: Dict[str, torch.Tensor], indices: torch.Tensor, weights: torch.Tensor) -> float:
        """ Performs learning update on a batch, calculates combined loss, returns avg loss scalar. """
        self.train() # Set online networks to training mode
        self.set_target_networks_eval() # Ensure targets are in eval mode

        try: # Wrap main logic in try block for better error handling
            states = batch_data['states']; rewards = batch_data['rewards'];
            next_states = batch_data['next_states']; dones = batch_data['dones']

            # *** FIX: Get batch_size reliably from input ***
            batch_size = states.shape[0]
            # *** END FIX ***

            # --- Online Network Pass (for V(s) and metrics for loss) ---
            online_out = self.forward_batch(states, rewards, None, use_target_net=False)
            # Check if forward pass returned valid outputs
            if online_out is None or online_out.value_pred is None:
                 logger.error("Online forward pass failed in learn method. Skipping update.")
                 return 0.0 # Or some error indicator

            q_values_s = online_out.value_pred # V(s)

            # --- Target Network Pass (for V(s')) ---
            with torch.no_grad():
                 target_out = self.forward_batch(next_states, torch.zeros_like(rewards), None, use_target_net=True)
                 if target_out is None or target_out.value_pred is None:
                      logger.error("Target forward pass failed in learn method. Skipping update.")
                      return 0.0 # Or some error indicator
                 q_values_sp = target_out.value_pred # V(s')
                 q_targets = rewards + Config.RL.GAMMA * q_values_sp * (~dones) # Bellman target

            # --- Value Loss (TD Error) ---
            td_error = q_targets - q_values_s # Shape: [B, 1]
            if not is_safe(td_error):
                 logger.warning("Unsafe TD Error calculated. Clamping and continuing.")
                 td_error = torch.nan_to_num(td_error, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10, 10) # Clamp to reasonable range

            value_loss = F.smooth_l1_loss(q_values_s, q_targets.detach(), reduction='none') # Shape: [B, 1]
            weighted_value_loss = (weights * value_loss).mean() * Config.RL.VALUE_LOSS_WEIGHT

            # --- Syntrometric/RIH/HET Loss Components ---
            integration_I = online_out.integration_I; rho_score = online_out.rho_score; tau_t = online_out.tau_t
            stability = online_out.stability; lyapunov_proxy = online_out.lyapunov_max_proxy; zeta = online_out.zeta
            R_norm = torch.norm(online_out.R, dim=1, keepdim=True) if online_out.R is not None else torch.zeros_like(rewards) # Handle potential None
            self_consistency = online_out.self_consistency
            # Ensure telezentrik_t is tensor [B, 1] or scalar compatible with batch ops
            telezentrik_t = online_out.telezentrik if online_out.telezentrik is not None else torch.zeros_like(rewards)
            if telezentrik_t.ndim == 0: telezentrik_t = telezentrik_t.repeat(batch_size).unsqueeze(1)
            elif telezentrik_t.ndim == 1: telezentrik_t = telezentrik_t.unsqueeze(1)

            # Ensure all components are tensors with batch dimension before combining
            rih_integration_penalty = F.relu(tau_t - integration_I) if tau_t is not None and integration_I is not None else torch.zeros_like(rewards)
            rih_reflexivity_penalty = F.relu(Config.RL.RHO_SIMILARITY_THRESHOLD - rho_score) if rho_score is not None else torch.zeros_like(rewards)
            stability_penalty = F.relu(stability - self.selector.detach())**2 if stability is not None else torch.zeros_like(rewards)
            dynamical_stability_penalty = F.relu(lyapunov_proxy) if lyapunov_proxy is not None else torch.zeros_like(rewards)
            geometric_coherence_penalty = torch.tensor(0.0, device=DEVICE) # Placeholder
            telezentrik_reward = telezentrik_t # Already shaped [B, 1]
            consistency_reward = self_consistency if self_consistency is not None else torch.zeros_like(rewards)
            complexity_penalty = torch.abs(zeta) if zeta is not None else torch.zeros_like(rewards)
            curvature_penalty = R_norm # Already shaped [B, 1]

            # Combine weighted terms
            syntrometric_loss_terms = (
                Config.RL.INTEGRATION_WEIGHT * rih_integration_penalty +
                Config.RL.REFLEXIVITY_WEIGHT * rih_reflexivity_penalty +
                Config.RL.STABILITY_WEIGHT * stability_penalty +
                Config.RL.DYNAMICAL_STABILITY_WEIGHT * dynamical_stability_penalty +
                Config.RL.GEOMETRIC_COHERENCE_WEIGHT * geometric_coherence_penalty +
                Config.RL.COMPLEXITY_WEIGHT * complexity_penalty +
                Config.RL.CURVATURE_WEIGHT * curvature_penalty -
                Config.RL.TELEZENTRIK_WEIGHT * telezentrik_reward -
                Config.RL.CONSISTENCY_WEIGHT * consistency_reward
            )
            syntrometric_loss = syntrometric_loss_terms.mean() # Average over batch

            # --- Total Loss ---
            total_loss = weighted_value_loss + syntrometric_loss

            # --- Optimization ---
            self.optimizer.zero_grad()
            loss_val = 0.0 # Default loss value for logging if step skipped
            if is_safe(total_loss) and total_loss.requires_grad:
                 total_loss.backward()
                 torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=Config.RL.GRADIENT_CLIP_AGENT)
                 self.optimizer.step()
                 loss_val = total_loss.item()
            else: logger.warning(f"Skipping opt step. Loss unsafe or no grad: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss}")

            # --- PER Priority Update ---
            priorities = td_error.abs().detach().squeeze() + 1e-5
            self.memory.update_priorities(indices, priorities)

            # --- Target Network Soft Update ---
            if self.step_count % Config.RL.TARGET_NETWORK_UPDATE_FREQ == 0: self.soft_update_target_networks()

            # --- Add experience to memory AFTER gradients are computed ---
            # Use states/rewards/etc from the *start* of the learn step
            beliefs_for_mem = online_out.belief_embedding # [B, H]
            # Ensure beliefs_for_mem is valid before looping
            if beliefs_for_mem is not None and beliefs_for_mem.shape[0] == batch_size:
                for i in range(batch_size): # Now batch_size is defined
                    exp = Experience(
                        states[i].detach(), beliefs_for_mem[i].detach(), rewards[i].item(),
                        next_states[i].detach(), dones[i].item(),
                        td_error[i].item() # Store original TD error for this sample
                    )
                    self.memory.add(exp) # Add individual experiences
            else:
                 logger.warning("Skipping memory add in learn step due to invalid belief embeddings.")

            self.step_count += 1; self.beta = min(1.0, self.beta + self.beta_increment)
            return loss_val

        except Exception as learn_err:
            logger.error(f"Error during agent learn step: {learn_err}", exc_info=True)
            return 0.0 # Return 0 loss on error to avoid breaking orchestrator

    # === Single Instance Forward (Wrapper) ===
    def forward_single(self, state: torch.Tensor, reward: float, state_history: torch.Tensor) -> AgentForwardReturnType:
        """ Runs forward pass for a single state instance. """
        self.eval()
        state_batch = state.unsqueeze(0); reward_batch = torch.tensor([[reward]], device=DEVICE)
        history_batch = state_history.unsqueeze(0) if state_history.ndim == 2 else state_history # Add batch dim if needed
        with torch.no_grad():
             batch_out = self.forward_batch(state_batch, reward_batch, history_batch, use_target_net=False)
        self.train() # Return to train mode
        return batch_out

    # === Generate Response (Contextual) ===
    def generate_response(self, context: str, att_score: float) -> str:
        """ Generates text response using TransformerGPT, prepending internal state. """
        if not self.gpt: return "[GPT Error: Not Initialized]"
        emo_summary = "neutral"
        if hasattr(self, 'prev_emotions') and self.prev_emotions is not None:
            try: emo_cpu = self.prev_emotions.cpu().numpy(); emo_summary = f"feeling({emo_cpu.round(2)})"
            except Exception: pass
        state_context = f"[Internal: {emo_summary}, Att={att_score:.2f}] "
        full_prompt = state_context + "\n" + context
        temp = Config.NLP.GPT_TEMPERATURE * (1.0 + (1.0 - max(0, min(1, att_score)))*0.2) # Clamp att_score
        return self.gpt.generate(full_prompt, temperature=temp, top_p=Config.NLP.GPT_TOP_P)

    # === Save/Load (Includes ALL target nets and GNN modules) ===
    def save_state(self, agent_path=AGENT_SAVE_PATH, gpt_path=GPT_SAVE_PATH, optimizer_path=OPTIMIZER_SAVE_PATH, target_path_suffix=TARGET_NET_SAVE_SUFFIX):
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
                'prev_emotions': self.prev_emotions,
                'prev_belief_embedding': self.prev_belief_embedding,
                'selector': self.selector,
                'step_count': self.step_count, 'beta': self.beta,
                'state_dim': self.state_dim, 'hidden_dim': self.hidden_dim, 'gnn_hidden_dim': self.gnn_hidden_dim,
            }
            torch.save(agent_state, agent_path)
            # Save Target Networks separately
            torch.save(self.target_encoder.state_dict(), agent_path + target_path_suffix + "_encoder")
            torch.save(self.target_gnn_layers_module.state_dict(), agent_path + target_path_suffix + "_gnn")
            torch.save(self.target_self_reflect_layer.state_dict(), agent_path + target_path_suffix + "_reflect")
            torch.save(self.target_value_head.state_dict(), agent_path + target_path_suffix + "_value")
            logger.debug(f"Core agent and target states saved.")

            self.gpt.save_model(gpt_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            logger.info("Agent, Target Nets, GPT, and optimizer states saved successfully.")
        except Exception as e: logger.error(f"Error saving agent state: {e}", exc_info=True)

    def load_state(self, agent_path=AGENT_SAVE_PATH, gpt_path=GPT_SAVE_PATH, optimizer_path=OPTIMIZER_SAVE_PATH, target_path_suffix=TARGET_NET_SAVE_SUFFIX) -> bool:
        logger.info(f"Loading agent state from {agent_path}...")
        if not os.path.exists(agent_path): logger.error(f"Agent state file not found: {agent_path}."); return False
        try:
            agent_state = torch.load(agent_path, map_location=DEVICE)
            # Dimension Validation
            if agent_state.get('state_dim') != self.state_dim or \
               agent_state.get('hidden_dim') != self.hidden_dim or \
               agent_state.get('gnn_hidden_dim') != self.gnn_hidden_dim:
                 logger.critical(f"CRITICAL LOAD ERROR: Dimension mismatch."); return False

            self.encoder.load_state_dict(agent_state['encoder_state_dict'])
            self.gnn_layers_module.load_state_dict(agent_state['gnn_layers_module_state_dict'])
            self.emotional_module.load_state_dict(agent_state['emotional_module_state_dict'])
            self.self_reflect_layer.load_state_dict(agent_state['self_reflect_layer_state_dict'])
            self.feedback.load_state_dict(agent_state['feedback_state_dict'])
            self.qualia_output_head.load_state_dict(agent_state['qualia_output_head_state_dict'])
            self.value_head.load_state_dict(agent_state['value_head_state_dict'])

            # Load Target Networks safely
            target_paths = { "encoder": agent_path + target_path_suffix + "_encoder", "gnn": agent_path + target_path_suffix + "_gnn", "reflect": agent_path + target_path_suffix + "_reflect", "value": agent_path + target_path_suffix + "_value", }
            online_nets = { "encoder": self.encoder, "gnn": self.gnn_layers_module, "reflect": self.self_reflect_layer, "value": self.value_head, }
            target_nets = { "encoder": self.target_encoder, "gnn": self.target_gnn_layers_module, "reflect": self.target_self_reflect_layer, "value": self.target_value_head, }
            for name, path in target_paths.items():
                 if os.path.exists(path): target_nets[name].load_state_dict(torch.load(path, map_location=DEVICE))
                 else: logger.warning(f"Target {name} state not found, copying from online."); target_nets[name].load_state_dict(online_nets[name].state_dict())
            self.set_target_networks_eval()

            # Load Metadata
            self.prev_emotions = agent_state.get('prev_emotions', self.prev_emotions).to(DEVICE)
            self.prev_belief_embedding = agent_state.get('prev_belief_embedding', self.prev_belief_embedding)
            if self.prev_belief_embedding is not None: self.prev_belief_embedding = self.prev_belief_embedding.to(DEVICE)
            self.selector = agent_state.get('selector', self.selector).to(DEVICE)
            self.step_count = agent_state.get('step_count', self.step_count); self.beta = agent_state.get('beta', self.beta)
            logger.info("Core agent & target networks and metadata loaded.")

            if not self.gpt.load_model(gpt_path): logger.warning(f"GPT load failed from {gpt_path}."); self.gpt = TransformerGPT() # Re-init base

            if os.path.exists(optimizer_path):
                 self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
                 for group in self.optimizer.param_groups: group['lr'] = Config.RL.LR
                 logger.info("Optimizer state loaded.")
            else: logger.warning(f"Optimizer state file not found: {optimizer_path}.")

            self.eval() # Set online networks to eval mode after loading
            logger.info("Agent state loading complete.")
            return True
        except FileNotFoundError: logger.error(f"Error: State file not found at {agent_path}."); return False
        except KeyError as e: logger.error(f"Error loading state: Missing key {e}.", exc_info=True); return False
        except RuntimeError as e: logger.error(f"RuntimeError loading state dict: {e}.", exc_info=True); return False
        except Exception as e: logger.error(f"Unexpected error loading agent state: {e}", exc_info=True); return False

# --- END OF FILE agent.py ---

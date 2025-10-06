import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class SAEConfig:
    d_emb: int
    d_hidden: int
    top_k: int
    up_proj_bias: bool
    device: str
    dtype: torch.dtype

class SAE(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # --- your original layers & tying ---
        self.up_proj = nn.Linear(
            config.d_emb, config.d_hidden,
            dtype=config.dtype, device=config.device, bias=config.up_proj_bias
        )
        self.down_proj = nn.Linear(
            config.d_hidden, config.d_emb,
            dtype=config.dtype, device=config.device, bias=False
        )
        # keep your original cloned tie semantics
        self.down_proj.weight = nn.Parameter(self.up_proj.weight.T.clone())

        # --- your original counter (kept exactly) ---
        self.feature_activation_counter = torch.zeros(
            config.d_hidden, dtype=torch.int32, device=config.device
        )
        self.reset_counters()

        # --- added: activation logging state (optional) ---
        self._log_acts = False
        self.register_buffer("_act_sum",   torch.zeros(config.d_hidden, dtype=torch.float32))
        self.register_buffer("_act_hits",  torch.zeros(config.d_hidden, dtype=torch.float32))
        self.register_buffer("_n_samples", torch.tensor(0, dtype=torch.long))
        self._last_acts = None  # last batch activations (detached)

    # ---- your original API (unchanged) ----
    def reset_counters(self):
        old = self.feature_activation_counter.detach().cpu().numpy()
        torch.zero_(self.feature_activation_counter)
        return old

    # ---- internal encode/decode to mirror your training path ----
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.up_proj(x)
        z = F.relu(z)

        k = self.config.top_k
        d = z.size(-1)
        if not (0 < k <= d):
            raise ValueError(f"top_k must be in [1, {d}], got {k}")

        # top-k gate (MPS-friendly)
        k = min(k, d)
        vals, idxs = torch.topk(z, k=k, dim=-1)
        mask = torch.zeros_like(z, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idxs, value=True)

        # strict > kth-from-top (avoid overselecting on ties)
        # kth value per row:
        kth = vals[..., -1].unsqueeze(-1)
        gt_mask = z > kth
        mask = mask & gt_mask

        z = torch.where(mask, z, z.new_zeros(()))

        # your feature activation counter (cast to int32 to match buffer)
        self.feature_activation_counter += mask.sum(dim=0).to(self.feature_activation_counter.dtype)

        return z

    def _decode(self, a: torch.Tensor) -> torch.Tensor:
        return self.down_proj(a)

    def forward(self, embs: torch.Tensor, return_acts: bool = False):
        a = self._encode(embs)

        # optional activation logging
        if self._log_acts:
            a_det = a.detach()
            self._act_sum  += a_det.sum(dim=0).to(self._act_sum.dtype)
            self._act_hits += (a_det > 0).sum(dim=0).to(self._act_hits.dtype)
            self._n_samples += a_det.shape[0]
            self._last_acts = a_det

        x_hat = self._decode(a)
        return (x_hat, a) if return_acts else x_hat

    # ---- added: activation logging utilities (non-breaking) ----
    def reset_activation_stats(self):
        self._act_sum.zero_()
        self._act_hits.zero_()
        self._n_samples.zero_()
        self._last_acts = None

    def get_activation_stats(self):
        n = int(self._n_samples.item())
        if n == 0:
            d = self._act_sum.shape[0]
            dev = self._act_sum.device
            return (torch.zeros(d, device=dev),
                    torch.zeros(d, device=dev),
                    0,
                    self._last_acts)
        mean_mag = self._act_sum / float(n)   # average magnitude incl. zeros
        freq     = self._act_hits / float(n)  # fraction > 0
        return mean_mag, freq, n, self._last_acts

    @contextmanager
    def activation_logging(self, enabled: bool = True, reset: bool = True):
        prev = self._log_acts
        try:
            if enabled:
                if reset:
                    self.reset_activation_stats()
                self._log_acts = True
            yield
        finally:
            self._log_acts = prev

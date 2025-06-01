import torch
import torch.nn as nn
from .helper import PositionalEncoding, SelfAttnBlock, DiffusionRefiner, CrossAttnBlock

# ------- custom ts decoder_layers -------
class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for time series reconstruction.
    Optionally applies a diffusion denoising tail.

    Args:
        ts_dim (int): Output time series dimension.
        output_dim (int): Latent/embedding dimension.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout rate.
        project_input (bool): If True, project input to ts_dim.
        diffusion_steps (int): If >0, apply diffusion tail.
        diff_txt_proj (bool): If True, project text embedding in diffusion tail.
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        project_input: bool = False,
        diffusion_steps = 0,
        diff_txt_proj = True,
    ):
        super().__init__()
        hidden_dim = output_dim

        self.project_input = project_input
        if project_input:
            self.input_projection = nn.Linear(output_dim, ts_dim)
            hidden_dim = ts_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, ts_dim)
        # self.output_ln = nn.LayerNorm(hidden_dim)
        # self.output_ffn = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 4),  # Expand dimension
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim * 4, ts_dim)  # Project to final dimension
        # )
        
        # Optional diffusion tail
        self.diffusion_steps = diffusion_steps
        if self.diffusion_steps > 0:
            self.diffusion_tail = DiffusionRefiner(
                ts_dim=ts_dim,
                txt_dim=hidden_dim,
                n_steps=self.diffusion_steps,
                diff_txt_proj=diff_txt_proj
            )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 3D: [B, L, E]."""
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts_emb: [B, E] or [B, 1, E] - time series embedding (target)
            txt_emb: [B, E] or [B, 1, E] - text embedding (memory)
        Returns:
            ts_hat: [B, ts_dim] - reconstructed time series
        """
        tgt = self._as_sequence(ts_emb)
        memory = self._as_sequence(txt_emb)
        if self.project_input:
            tgt = self.input_projection(tgt)
            memory = self.input_projection(memory)
        tgt = self.pos_encoder(tgt)
        memory = self.pos_encoder(memory)
        dec_out = self.decoder(tgt=tgt, memory=memory)
        ts_hat = self.output_projection(dec_out).squeeze(1)
        if self.diffusion_steps > 0:
            ts_hat = self.diffusion_tail(ts_hat, txt_emb)
        return ts_hat


class SelfAttnDecoder(nn.Module):
    """
    Decoder using stacked self-attention blocks and optional diffusion tail.

    Args:
        ts_dim (int): Output time series dimension.
        output_dim (int): Latent/embedding dimension.
        hidden_dim (int, optional): Internal dimension. Defaults to output_dim.
        nhead (int): Number of attention heads.
        num_layers (int): Number of self-attention layers.
        diffusion_steps (int): If >0, apply diffusion tail.
        diff_txt_proj (bool): If True, project text embedding in diffusion tail.
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        nhead: int = 8,
        num_layers: int = 4,
        ffn_mult: int = 1,
        diffusion_steps: int = 0,
        diff_txt_proj: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim or output_dim
        if hidden_dim is not None and hidden_dim != output_dim:
            self.proj_ts = nn.Linear(output_dim, self.hidden_dim)
            self.proj_text = nn.Linear(output_dim, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.blocks = nn.Sequential(
            *[
                SelfAttnBlock(
                    width=self.hidden_dim,
                    heads=nhead,
                    drop=0.0,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, ts_dim),
        )
        self.diffusion_steps = diffusion_steps
        if diffusion_steps > 0:
            self.diffusion_tail = DiffusionRefiner(
                ts_dim=ts_dim,
                txt_dim=output_dim,
                n_steps=diffusion_steps,
                diff_txt_proj=diff_txt_proj
            )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 3D: [B, L, E]."""
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts_emb: [B, E] or [B, 1, E] - time series embedding
            txt_emb: [B, E] or [B, 1, E] - text embedding
        Returns:
            ts_hat: [B, ts_dim] - reconstructed time series
        """
        if hasattr(self, "proj_ts"):
            tgt = self.proj_ts(ts_emb)
            memory = self.proj_text(txt_emb)
        else:
            tgt, memory = ts_emb, txt_emb
        tgt = self._as_sequence(tgt)
        memory = self._as_sequence(memory)
        tokens = torch.cat([tgt, memory], dim=1)
        tokens = self.pos_encoder(tokens)
        h = self.blocks(tokens)
        ts_hat = self.out(h[:, 0])
        if self.diffusion_steps > 0:
            ts_hat = self.diffusion_tail(ts_hat, txt_emb)
        return ts_hat


class CrossAttnDecoder(nn.Module):
    """
    Decoder using stacked cross-attention blocks and optional diffusion tail.

    Args:
        ts_dim (int): Output time series dimension.
        output_dim (int): Latent/embedding dimension.
        hidden_dim (int, optional): Internal dimension. Defaults to output_dim.
        nhead (int): Number of attention heads.
        num_layers (int): Number of cross-attention layers.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout rate.
        diffusion_steps (int): If >0, apply diffusion tail.
        diff_txt_proj (bool): If True, project text embedding in diffusion tail.
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        diffusion_steps: int = 0,
        diff_txt_proj: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim or output_dim
        if hidden_dim is not None and hidden_dim != output_dim:
            self.proj_ts = nn.Linear(output_dim, self.hidden_dim)
            self.proj_text = nn.Linear(output_dim, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.blocks = nn.Sequential(
            *[CrossAttnBlock(self.hidden_dim, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, ts_dim),
        )
        self.diffusion_steps = diffusion_steps
        if diffusion_steps > 0:
            self.diffusion_tail = DiffusionRefiner(
                ts_dim=ts_dim,
                txt_dim=output_dim,
                n_steps=diffusion_steps,
                diff_txt_proj=diff_txt_proj
            )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 3D: [B, L, E]."""
        return x.unsqueeze(1) if x.dim() == 2 else x
    
    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts_emb: [B, E] or [B, 1, E] - time series embedding
            txt_emb: [B, E] or [B, 1, E] - text embedding
        Returns:
            ts_hat: [B, ts_dim] - reconstructed time series
        """
        if hasattr(self, "proj_ts"):
            tgt, memory = self.proj_ts(ts_emb), self.proj_text(txt_emb)
        else:
            tgt, memory = ts_emb, txt_emb
        tgt = self._as_sequence(tgt)
        memory = self._as_sequence(memory)
        tgt = self.pos_encoder(tgt)
        memory = self.pos_encoder(memory)
        for blk in self.blocks:
            tgt = blk(tgt, memory)
        ts_hat = self.out(tgt.squeeze(1))
        if self.diffusion_steps > 0:
            ts_hat = self.diffusion_tail(ts_hat, txt_emb)
        return ts_hat


class BiCrossAttnDecoder(nn.Module):
    """
    Bidirectional cross-attention decoder for time series reconstruction.
    Each layer alternates TS→TXT and TXT→TS attention.
    Optionally applies a diffusion denoising tail.

    Args:
        ts_dim (int): Output time series dimension.
        output_dim (int): Latent/embedding dimension.
        hidden_dim (int, optional): Internal dimension. Defaults to output_dim.
        nhead (int): Number of attention heads.
        num_layers (int): Number of bidirectional attention layers.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout rate.
        diffusion_steps (int): If >0, apply diffusion tail.
        diff_txt_proj (bool): If True, project text embedding in diffusion tail.
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        diffusion_steps: int = 0,
        diff_txt_proj: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim or output_dim
        if hidden_dim is not None and hidden_dim != output_dim:
            self.proj_ts = nn.Linear(output_dim, self.hidden_dim)
            self.proj_text = nn.Linear(output_dim, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ts2txt": CrossAttnBlock(self.hidden_dim, nhead, dim_feedforward, dropout),
                "txt2ts": CrossAttnBlock(self.hidden_dim, nhead, dim_feedforward, dropout),
            }) for _ in range(num_layers)
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, ts_dim),
        )
        self.diffusion_steps = diffusion_steps
        if diffusion_steps > 0:
            self.diffusion_tail = DiffusionRefiner(
                ts_dim=ts_dim,
                txt_dim=output_dim,
                n_steps=diffusion_steps,
                diff_txt_proj=diff_txt_proj
            )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 3D: [B, L, E]."""
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts_emb: [B, E] or [B, 1, E] - time series embedding
            txt_emb: [B, E] or [B, 1, E] - text embedding
        Returns:
            ts_hat: [B, ts_dim] - reconstructed time series
        """
        if hasattr(self, "proj_ts"):
            tgt, mem = self.proj_ts(ts_emb), self.proj_text(txt_emb)
        else:
            tgt, mem = ts_emb, txt_emb
        tgt = self._as_sequence(tgt)
        mem = self._as_sequence(mem)
        tgt = self.pos_encoder(tgt)
        mem = self.pos_encoder(mem)
        for layer in self.layers:
            tgt = layer["ts2txt"](tgt, mem)   # TS attends to TXT
            mem = layer["txt2ts"](mem, tgt)   # TXT attends back to TSs
        ts_hat = self.out(tgt.squeeze(1))     # [B, ts_dim]

        if self.diffusion_steps > 0:
            ts_hat = self.diffusion_tail(ts_hat, txt_emb)
        return ts_hat

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x):
        seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Original cos/sin shape: [seq_len, head_dim]
    # q/k shape: [batch_size, n_heads, seq_len, head_dim]
    # We need cos/sin to be [1, 1, seq_len, head_dim] for proper broadcasting
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DiffusionStepEmbedding(nn.Module):
    """Diffusion-style step embedding for iterative latent refinement."""
    def __init__(self, dim, max_steps=32):
        super().__init__()
        self.step_emb = nn.Embedding(max_steps, dim)
        self.project = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, step_index):
        if step_index.dim() == 0:
            step_index = step_index.unsqueeze(0)
        emb = self.step_emb(step_index)
        return self.project(emb)


class NumericalProjector(nn.Module):
    """Project numeric features into the latent dimension."""
    def __init__(self, dim, input_features=1, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(1, dim // 2)
        self.projector = nn.Sequential(
            nn.Linear(input_features, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, numerical_values, numerical_mask=None):
        if numerical_values is None:
            return None
        if numerical_values.dim() == 2:
            numerical_values = numerical_values.unsqueeze(-1)
        num_emb = self.projector(numerical_values.float())
        if numerical_mask is not None:
            numerical_mask = numerical_mask.unsqueeze(-1).to(dtype=num_emb.dtype)
            num_emb = num_emb * numerical_mask
        return num_emb


class SwiGLU(nn.Module):
    """SwiGLU activation function with optional dropout"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE"""
    def __init__(self, dim, n_heads, max_seq_len=512, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    def __init__(self, dim, n_heads, mlp_ratio=4, max_seq_len=512, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# Tiny Recursive Model
# ============================================================================

class TinyRecursiveNetwork(nn.Module):
    """
    The core tiny network used in TRM.
    Only 2 layers as per the paper's finding that smaller is better.
    """
    def __init__(self, dim, n_heads=8, n_layers=2, mlp_ratio=4, max_seq_len=512, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, max_seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model for Text Generation

    Architecture based on TRM paper:
    - Wide, shallow base network for stable recursion
    - Diffusion-inspired latent denoising
    - Numerical injection for finance reasoning

    For text generation:
    - x: embedded input sequence (context)
    - y: current token predictions (embedded)
    - z: latent reasoning state
    """
    def __init__(
        self,
        vocab_size,
        dim=768,
        n_heads=12,
        n_layers=4,
        mlp_ratio=4,
        max_seq_len=256,
        n_latent_recursions=6,
        n_improvement_cycles=3,
        dropout=0.1,
        tie_embeddings=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_latent_recursions = n_latent_recursions
        self.n_improvement_cycles = n_improvement_cycles
        self.noise_base = 0.08

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.emb_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # Wide, shallow recursive network
        self.net = TinyRecursiveNetwork(dim, n_heads, n_layers, mlp_ratio, max_seq_len, dropout=dropout)

        # Diffusion-inspired conditioning
        self.step_embed = DiffusionStepEmbedding(dim, max_steps=32)
        self.z_update = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

        # Numerical injection
        self.numerical_projector = NumericalProjector(dim)
        self.numerical_fuser = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

        # Projection layers for combining x, y, z
        self.combine_xyz = nn.Linear(dim * 3, dim, bias=False)
        self.combine_yz = nn.Linear(dim * 2, dim, bias=False)

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Halting head for ACT (simplified - no Q-learning)
        self.halt_head = nn.Linear(dim, 1, bias=False)

        # Learnable initial states for y and z
        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.tie_embeddings = tie_embeddings
        self.use_checkpoint = use_checkpoint

        self._init_weights()

        # Optionally tie embedding and output weights (helps sample efficiency)
        if self.tie_embeddings:
            try:
                self.output_head.weight = self.token_emb.weight
            except Exception:
                pass

        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Built TinyRecursiveModel with {params:,} trainable parameters")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_embeddings(self, input_ids):
        """Get token + position embeddings"""
        B, T = input_ids.shape
        # Clamp input_ids to valid range
        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        # Clamp position to max_seq_len
        T = min(T, self.max_seq_len)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.emb_dropout(self.token_emb(input_ids[:, :T]) + self.pos_emb(pos))

    def latent_recursion(self, x, y, z, step, numerical_values=None, numerical_mask=None):
        """
        Single recursion cycle using diffusion-inspired latent denoising.
        """
        step_id = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
        step_emb = self.step_embed(step_id).unsqueeze(1).expand(-1, x.size(1), -1)

        if self.training and step < self.n_latent_recursions:
            noise_scale = self.noise_base * (1.0 - step / max(1, self.n_latent_recursions))
            z = z + torch.randn_like(z) * noise_scale

        combined = self.combine_xyz(torch.cat([x, y, z], dim=-1)) + step_emb
        z_pred = self.net(combined)
        z = z + self.z_update(z_pred)

        if numerical_values is not None:
            num_emb = self.numerical_projector(numerical_values, numerical_mask)
            if num_emb is not None:
                z = z + self.numerical_fuser(torch.cat([z, num_emb], dim=-1))

        combined_yz = self.combine_yz(torch.cat([y, z], dim=-1)) + step_emb
        y = self.net(combined_yz)
        return y, z

    def deep_recursion(self, x, y, z, use_grad=True, numerical_values=None, numerical_mask=None):
        """
        Deep recursion with T improvement cycles.
        First T-1 cycles without gradients, last cycle with gradients.
        """
        if not use_grad:
            with torch.no_grad():
                for step in range(self.n_improvement_cycles):
                    y, z = self.latent_recursion(
                        x,
                        y,
                        z,
                        step,
                        numerical_values=numerical_values,
                        numerical_mask=numerical_mask,
                    )
            return y.detach(), z.detach()

        with torch.no_grad():
            for step in range(self.n_improvement_cycles - 1):
                y, z = self.latent_recursion(
                    x,
                    y,
                    z,
                    step,
                    numerical_values=numerical_values,
                    numerical_mask=numerical_mask,
                )

        y, z = self.latent_recursion(
            x,
            y,
            z,
            self.n_improvement_cycles - 1,
            numerical_values=numerical_values,
            numerical_mask=numerical_mask,
        )

        return y.detach(), z.detach(), self.output_head(y), self.halt_head(y.mean(dim=1))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        targets=None,
        n_supervision_steps=4,
        numerical_values=None,
        numerical_mask=None,
        **kwargs,
    ):
        """
        Forward pass with deep supervision.

        Args:
            input_ids: [B, T] input token IDs
            attention_mask: optional [B, T] attention mask
            targets: [B, T] target token IDs (for training)
            numerical_values: optional [B, T] or [B, T, K] numeric features
            numerical_mask: optional [B, T] boolean mask for numeric values
            n_supervision_steps: number of deep supervision steps

        Returns:
            If training: loss
            If inference: object with logits
        """
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T].clamp(0, self.vocab_size - 1)

        x = self.get_embeddings(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :T].unsqueeze(-1).to(dtype=x.dtype, device=x.device)
            x = x * attention_mask

        if numerical_values is not None:
            numerical_values = numerical_values[:, :T]
            if numerical_mask is not None:
                numerical_mask = numerical_mask[:, :T]

        # Initialize y and z
        y = self.y_init.expand(B, T, -1).clone().to(dtype=x.dtype, device=x.device)
        z = self.z_init.expand(B, T, -1).clone().to(dtype=x.dtype, device=x.device)

        if targets is None:
            y, z = self.deep_recursion(
                x,
                y,
                z,
                use_grad=False,
                numerical_values=numerical_values,
                numerical_mask=numerical_mask,
            )
            return SimpleNamespace(logits=self.output_head(y))

        targets = targets[:, :T].clamp(0, self.vocab_size - 1)
        total_loss = 0.0

        for step in range(n_supervision_steps):
            y, z, logits, halt_logit = self.deep_recursion(
                x,
                y,
                z,
                use_grad=True,
                numerical_values=numerical_values,
                numerical_mask=numerical_mask,
            )

            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (targets != -100)
                correct = ((preds == targets) & mask).float().sum() / mask.float().sum().clamp(min=1)
            halt_loss = F.binary_cross_entropy_with_logits(
                halt_logit.squeeze(-1),
                correct.expand(B),
            )

            total_loss = total_loss + ce_loss + 0.1 * halt_loss

        return total_loss / n_supervision_steps

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=None,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
        pad_token_id=None,
        do_sample=False,
        stopping_criteria=None,
        **kwargs,
    ):
        """Generate text autoregressively with HuggingFace-style args."""
        self.eval()

        if max_length is None:
            max_length = input_ids.shape[1] + max_new_tokens
        max_length = min(max_length, self.max_seq_len)

        for _ in range(max_length - input_ids.shape[1]):
            # Crop to max_seq_len - 1 to leave room for prediction
            idx_cond = input_ids[:, -(self.max_seq_len - 1):]

            # Clamp input ids to valid vocab range
            idx_cond = idx_cond.clamp(0, self.vocab_size - 1)

            # Get predictions
            output = self(idx_cond)
            logits = output.logits if hasattr(output, 'logits') else output
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if stopping_criteria is not None:
                stop = stopping_criteria(input_ids, logits)
                if isinstance(stop, torch.Tensor):
                    if stop.numel() == 1:
                        stop = bool(stop.item())
                    else:
                        stop = stop.all().item()
                if stop:
                    break

        return input_ids

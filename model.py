import torch
import torch.nn as nn
from transformers import BertModel
from config import Config

class DiffusionBERT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Initialize BERT model with trainable parameters
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # Spindle schedule parameters
        self.register_buffer('alpha', torch.zeros(config.diffusion_steps))
        self._init_spindle_schedule()
        
        # Projection layer for vocabulary
        self.proj = nn.Linear(768, 30522)  # BERT hidden_size=768, vocab_size=30522
    
    def _init_spindle_schedule(self):
        """Initialize the spindle noise schedule."""
        T = self.config.diffusion_steps
        lambda_val = self.config.lambda_val
        
        # Precompute alpha values for each step
        for t in range(1, T+1):
            self.alpha[t-1] = 1 - t/T - lambda_val * torch.sin(torch.tensor(t * torch.pi / T))
    
    def forward(self, x_t, t=None):
        """
        Predict x_{t-1} given x_t and optional time step t.
        Implements time-agnostic decoding when t is None.
        """
        # Time-agnostic decoding: infer t from mask ratio
        if t is None:
            mask_ratio = (x_t == 0).float().mean()  # Assuming 0 is [MASK] token
            t = torch.round((1 - mask_ratio) * self.config.diffusion_steps).long()
        
        # Ensure input is 2D [batch_size, seq_len]
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        
        # Ensure t is properly shaped
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
            
        # Get BERT embeddings with attention mask
        attention_mask = (x_t != 0).long()  # Create mask where 0 is padding
        
        # Explicitly get input shape
        input_shape = x_t.size()
        if len(input_shape) != 2:
            raise ValueError(f"Input shape must be 2D [batch_size, seq_len], got {input_shape}")
            
        outputs = self.bert(input_ids=x_t, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Project hidden states to vocabulary size
        return self.proj(hidden_states)
    
    def apply_noise(self, x_0, t):
        """Apply noise to x_0 at step t using spindle schedule."""
        alpha_t = self.alpha[t-1].to(x_0.device)
        mask_prob = 1 - alpha_t
        
        # Create mask with proper broadcasting
        # First expand mask_prob to match batch size
        mask_prob = mask_prob.expand(x_0.size(0))
        # Then create random mask with same shape as x_0
        mask = torch.rand_like(x_0.float()) < mask_prob.view(-1, 1)
        x_t = x_0.clone()
        x_t[mask] = 0  # 0 represents [MASK] token
        
        return x_t

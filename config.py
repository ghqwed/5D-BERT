from dataclasses import dataclass

@dataclass
class Config:
    # Diffusion process parameters
    diffusion_steps: int = 2048  # Must match pretrained model
    lambda_val: float = 0.7  # Adjusted spindle schedule parameter
    
    # Model architecture
    hidden_size: int = 768  # BERT hidden size
    num_attention_heads: int = 12  # BERT attention heads
    num_hidden_layers: int = 12  # BERT layers
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-6
    warmup_steps: int = 10000
    
    # Generation parameters
    max_seq_length: int = 128  # Maximum sequence length
    top_k: int = 50  # Reduced top-k sampling
    top_p: float = 0.95  # Higher nucleus sampling threshold
    temperature: float = 1.0  # Increased temperature for more diversity
    repetition_penalty: float = 1.5  # Stronger penalty for repeated tokens
    
    def __post_init__(self):
        # Validate config values
        assert self.diffusion_steps > 0
        assert 0 < self.lambda_val <= 1.0

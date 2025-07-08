import torch
from transformers import BertTokenizer
from model import DiffusionBERT
from config import Config

def generate(config: Config, model_path="diffusion_bert.pth"):
    # Initialize model and tokenizer
    model = DiffusionBERT(config)
    model.load_state_dict(torch.load(model_path))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Start with single token input
    input_ids = torch.tensor([[101]], dtype=torch.long)  # [CLS] token
    
    # Simplified generation - just do one forward pass
    with torch.no_grad():
        print(f"Input shape before model: {input_ids.shape}")
        pred = model(input_ids, torch.tensor([1], device=input_ids.device))
        print(f"Output shape from model: {pred.shape}")
        
        # Get top predicted tokens
        _, top_indices = torch.topk(pred, k=10, dim=-1)
        print(f"Top predicted tokens: {top_indices[0,0].tolist()}")
        
        # Just take first token prediction
        input_ids = top_indices[0,0,0].unsqueeze(0).unsqueeze(0)
        
        # Simple decoding
        token_ids = [input_ids.item()]
        generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"\nGenerated text: {generated_text}\n")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(token_ids)}")
        
        return generated_text

if __name__ == "__main__":
    config = Config()
    generate(config)

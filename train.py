import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import DiffusionBERT
from config import Config
from transformers import BertTokenizer

class TextDataset(Dataset):
    """Simple text dataset for demonstration"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding.input_ids.squeeze(0)

def train(config: Config):
    # Initialize model and tokenizer with retry and longer timeout
    try:
        model = DiffusionBERT(config)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                                timeout=30, 
                                                resume_download=True)
    except Exception as e:
        print(f"Failed to load BERT model: {e}")
        print("Please check your internet connection and try again")
        return
    
    # Sample training data (in practice, use real dataset)
    train_texts = ["This is a sample sentence.", "Diffusion models are powerful."]
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(10):  # Demo: 10 epochs
        for batch in train_loader:
            # Sample random diffusion step
            t = torch.randint(1, config.diffusion_steps, (batch.size(0),))
            
            # Apply noise
            x_t = model.apply_noise(batch, t)
            
            # Predict x_{t-1}
            pred = model(x_t, t)
            
            # Cross entropy loss
            loss = torch.nn.functional.cross_entropy(
                pred.view(-1, pred.size(-1)),  # [batch*seq_len, vocab_size]
                batch.view(-1)  # [batch*seq_len]
            )
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), "diffusion_bert.pth")

if __name__ == "__main__":
    config = Config()
    train(config)

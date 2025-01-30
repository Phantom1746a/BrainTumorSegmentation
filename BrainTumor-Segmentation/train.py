import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loading import BraTSDataset, get_data_paths
from model import UNet
from utils import DiceBCELoss
import config

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_paths = get_data_paths(config.TRAIN_DATA_DIR)
    val_paths = get_data_paths(config.VAL_DATA_DIR)
    
    train_dataset = BraTSDataset(train_paths, config.train_transform)
    val_dataset = BraTSDataset(val_paths, config.val_transform)
    
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item() * images.size(0)
        
        # Save best model
        avg_val_loss = val_loss / len(val_dataset)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}, Val Loss: {avg_val_loss:.4f}')

if __name__ == '__main__':
    train()
import torch
from torch.utils.data import DataLoader
from data_loading import BraTSDataset, get_data_paths
from model import UNet
import config
from utils import save_predictions

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    
    # Data
    test_paths = get_data_paths(config.TEST_DATA_DIR)
    test_dataset = BraTSDataset(test_paths, config.val_transform)
    test_loader = DataLoader(test_dataset, config.BATCH_SIZE, shuffle=False)
    
    # Inference
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            save_predictions(images, outputs, masks)

if __name__ == '__main__':
    test()
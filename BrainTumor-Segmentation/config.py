import yaml
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

# Load YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Dataset paths
TRAIN_DATA_DIR = config['data']['train_dir']
VAL_DATA_DIR = config['data']['val_dir']
TEST_DATA_DIR = config['data']['test_dir']

# Hyperparameters
BATCH_SIZE = config['training']['batch_size']
NUM_EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
MODEL_SAVE_PATH = config['training']['model_save_path']

# Data transforms
train_transform = transforms.Compose([
    globals()[transform](**kwargs) if isinstance(kwargs, dict) else globals()[transform]()
    for transform, kwargs in config['transforms']['train']
])

val_transform = transforms.Compose([
    globals()[transform](**kwargs) if isinstance(kwargs, dict) else globals()[transform]()
    for transform, kwargs in config['transforms']['val']
])

# Model configuration
IN_CHANNELS = config['model']['in_channels']
OUT_CHANNELS = config['model']['out_channels']

# Device configuration
DEVICE = config['device']

# Output configuration
PREDICTIONS_DIR = config['output']['predictions_dir']
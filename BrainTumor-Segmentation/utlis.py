import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=0.5, smooth=1e-6):
        super().__init__()
        self.weight = weight
        self.smooth = smooth
        self.bce = torch.nn.BCELoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return self.weight * bce + (1 - self.weight) * (1 - dice)

def save_predictions(images, preds, targets, save_dir='predictions/'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        if i == 0:
            plt.imshow(images[0].cpu().numpy(), cmap='gray')
            plt.title('Input')
        elif i == 1:
            plt.imshow(targets[0].cpu().numpy(), cmap='gray')
            plt.title('Ground Truth')
        else:
            plt.imshow(preds[0].cpu().numpy() > 0.5, cmap='gray')
            plt.title('Prediction')
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, f'pred_{np.random.randint(1000)}.png'))
    plt.close()
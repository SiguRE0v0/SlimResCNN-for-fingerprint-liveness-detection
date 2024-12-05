import torch
from tqdm import tqdm
import torch.nn.functional as F

@torch.inference_mode()
def validation(model, val_loader, device, classes):
    model.eval()
    num_val_batches = len(val_loader)
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=num_val_batches, desc='Validation round', unit='img', position=0, leave=False) as pbar:
            for images, labels in val_loader:
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)
                total += labels.size(0)

                logits = model(images)
                if classes != 1:
                    out = F.softmax(logits, dim=1)
                    _, pred = torch.max(out, 1)
                    correct += torch.eq(pred, labels).sum().item()
                else:
                    out = F.sigmoid(logits)
                    pred = out > 0.5
                    correct += (pred == labels).sum().item()
                pbar.update(images.shape[0])

    model.train()
    accuracy = correct / total
    return accuracy

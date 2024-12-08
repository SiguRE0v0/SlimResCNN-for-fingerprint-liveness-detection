import argparse
import logging
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from Utils.dataset import FingerDataset
from Model import SlimResCNN
from tqdm import tqdm

dir_img = "./data/testing"

def get_args():
    parser = argparse.ArgumentParser(description='Predicting model')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--size', '-s', type=int, default=160, help='Size of the patches after preprocess', dest='size')
    parser.add_argument('--load', '-m', type=str, default=None, help='Load .pth model')
    parser.add_argument('--classes', '-c', type=int, default=1, help='The classes of the output')
    return parser.parse_args()

def predict(
        model,
        device,
        test_set,
        classes
):

    model.eval()

    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

    # Start predict
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_set), desc=f'Predicting', position=0, leave=False, unit='img') as pbar:
            for batch in test_loader:
                images, labels = batch
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)
                total += labels.size(0)
                with torch.no_grad():
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
            accuracy = correct / total
    model.train()
    return accuracy


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')
    transform = transforms.Compose([
        transforms.RandomCrop((112, 112))
    ])
    test_set = FingerDataset(dir_img, img_size=args.size, transform=transform, augmentations=False)

    model = SlimResCNN(in_channels=1)
    model = model.to(device)

    if args.load is None:
        logging.error(f'No model loaded, check the path of .pth')
        sys.exit()

    state_dict = torch.load(args.load)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {args.load}')

    accuracy = predict(model=model, device=device, test_set=test_set, classes=args.classes)
    logging.info(f'Accuracy: {accuracy}')
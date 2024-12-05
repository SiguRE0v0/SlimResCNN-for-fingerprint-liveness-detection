import argparse
import logging
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import os
from Model import SlimResCNN
from Utils.dataset import FingerDataset
from Utils.evaluate import validation

def get_args():
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-2, help='Learning rate', dest='lr')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--size', '-s', type=int, default=160, help='Size of the images after preprocess', dest='size')
    parser.add_argument('--numval', '-n', type=int, default=1, help="The number of validation round in each epoch", dest='num_val')
    parser.add_argument('--load', '-m', type=str, default=False, help='Load .pth model')
    parser.add_argument('--classes', '-c', type=int, default=1, help='The number of classes of labels', dest='classes')
    return parser.parse_args()


dir_img = './data/training/'
dir_checkpoint = './checkpoints'
dir_test = './data/testing/'


def train_model(
        model,
        device,
        epochs,
        batch_size,
        learning_rate,
        val_percent,
        img_size,
        classes,
        save_checkpoint: bool = True
):
    # Create Dataset
    transform = transforms.Compose([
        transforms.RandomCrop((112, 112))
    ])
    dataset = FingerDataset(dir_img, img_size=img_size, transform=transform)

    # Split into train / validation set and create dataloader
    if args.num_val > 0:
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1)
    else:
        n_val = 0
        n_train = len(dataset)
        train_set = dataset
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        val_loader = None
        test_set = FingerDataset(dir_test, img_size=args.size, augmentations=False)

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Patch size:      {img_size}
            Output classes:  {classes}
        ''')

    # Set up the optimizer and the loss
    # optimizer = optim.AdamW(model.parameters(), learning_rate)
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    if classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200000, gamma=0.8)

    # Begin training
    global_step = 0
    avg_acc = []
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        train_acc = 0
        total = 0
        correct = 0
        num_val = args.num_val
        division_step = n_train if num_val == 0 else n_train / num_val
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', position=0, leave=False, unit='img') as pbar:
            for batch in train_loader:
                images, labels = batch
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device)
                optimizer.zero_grad(set_to_none=True)

                logits = model(images)

                # loss and accuracy in training
                total += images.size(0)
                if classes != 1:
                    loss = criterion(logits, labels)
                    out = F.softmax(logits, dim=1)
                    _, pred = torch.max(out, 1)
                    correct += torch.eq(pred, labels).sum().item()
                else:
                    loss = criterion(logits, labels.float())
                    out = F.sigmoid(logits)
                    pred = out > 0.5
                    correct += (pred == labels).sum().item()
                train_acc = correct / total

                # optimize
                loss.backward()
                optimizer.step()
                global_step += 1
                scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']
                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': loss.item(), 'learning rate': current_lr, 'train accuracy': train_acc})

                # Evaluation round during epoch
                if num_val > 0 and (total >= division_step or total == n_train):
                    division_step += division_step
                    acc = validation(model, val_loader, device, classes)
                    if len(avg_acc) == 5:
                        avg_acc.pop(0)
                    avg_acc.append(acc)

                    print('\nValidation accuracy: {}'.format(acc))
        # if num_val == 0:
        #     acc = predict(model=model, device=device, test_set=test_set)
        #     if len(avg_acc) == 5:
        #         avg_acc.pop(0)
        #     avg_acc.append(acc)
        #     if args.scheduler:
        #         scheduler.step(sum(avg_acc) / len(avg_acc))

        logging.info(f'Epoch:{epoch} | Average acc:{sum(avg_acc) / len(avg_acc)} | Validation acc:{acc} | Train acc:{train_acc}')

        # Epoch finished, save model
        if save_checkpoint:
            state_dict = model.state_dict()
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)
            torch.save(state_dict, os.path.join(dir_checkpoint,'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


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

    model = SlimResCNN(in_channels=1)
    model = model.to(device)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    torch.cuda.empty_cache()
    train_model(
        model = model,
        device =device,
        epochs = args.epochs,
        batch_size = args.batch_size,
        learning_rate = args.lr,
        val_percent = args.val,
        img_size = args.size,
        classes = args.classes,
        )

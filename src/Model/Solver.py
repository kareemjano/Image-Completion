from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import os
from src.dataset.util import visualize_torch

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, cuda=False,
                 optimizer=torch.optim.Adam, criterion=torch.nn.MSELoss(), batch_size=32,
                 target_center=[128,128], target_size=40, scheduler=None, patience=5, checkpoint_dir=None):
        print("Cuda is available") if cuda else print("Cuda is not avaliable")

        self.model = model.to('cuda') if cuda else model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.cuda = cuda
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=mp.cpu_count())
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=mp.cpu_count())
        self.target_center = target_center
        self.target_size = target_size
        self.scheduler = scheduler
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        self.epoch = 0
        self.writer = None
        self.bad_epoch = 0
        self.min_loss = np.inf

    def get_model_output(self, in_imgs):
        outputs = self.model(in_imgs.float())
        top = int(self.target_center[0] - (self.target_size / 2))
        left = int(self.target_center[1] - (self.target_size / 2))
        return outputs[:, :, top:top + self.target_size, left:left + self.target_size].float()

    def train_epoch(self, validate=False):
        dataloader = self.train_loader if not validate else self.valid_loader
        epoch = self.epoch
        avg_acc = 0
        running_loss = 0.0

        self.model.train()
        if validate:
            self.model.eval()

        with tqdm(dataloader, total = len(dataloader)) as epoch_pbar:
            for i, data in enumerate(epoch_pbar):
                # get the inputs; data is a list of [inputs, labels]
                _, in_imgs, target_imgs = data
                if self.cuda:
                    in_imgs, target_imgs = in_imgs.cuda(), target_imgs.cuda()
                # zero the parameter gradients
                if not validate:
                    self.optimizer.zero_grad()

                # forward + backward + optimize
                loss_each = self.criterion(self.get_model_output(in_imgs), target_imgs.float())

                loss = torch.mean(loss_each)
                if not validate:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.detach().item()
                samples_loss = loss_each.detach().cpu().view(loss_each.shape[0], np.product(loss_each.shape[1:])).mean(axis=-1)
                avg_acc += np.mean(np.isclose(samples_loss, 0, atol=0.005))

                if not validate:
                    desc = f'Epoch Train {epoch} - loss {running_loss / (i + 1):.4f}'
                else:
                    desc = f'Validate - loss {running_loss / (i + 1):.4f}'

                epoch_pbar.set_description(desc)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = avg_acc / len(dataloader)
        log_name = 'Training' if not validate else 'Validate'
        _, in_images, _ = next(iter(dataloader))
        output_imgs, _ = self.inference(loader=dataloader)

        if self.writer:
            # log scaler to Tensorboard
            self.writer.add_scalar(f'{log_name} loss', epoch_loss, epoch)
            self.writer.add_scalar(f'{log_name} Accuracy', epoch_acc, epoch)
            # log images to Tensorboard
            self.writer.add_figure(log_name, visualize_torch(output_imgs, gray=True), global_step=epoch)

        print(f'{log_name} Loss: {epoch_loss:.4f}. Acc: {epoch_acc:.4f}')

        if validate:
            if self.patience > 0:
                if epoch_loss > self.min_loss:
                    self.bad_epoch += 1
                elif epoch_loss < self.min_loss:
                    self.bad_epoch -= 1 if self.bad_epoch > 0 else 0

            if epoch_loss < self.min_loss:
                self.min_loss = epoch_loss
                if self.checkpoint_dir:
                    if not os.path.exists(self.checkpoint_dir):
                        os.mkdir(self.checkpoint_dir)
                    self.save(os.path.join(self.checkpoint_dir, "checkpoint.model"), inference=False)

        return epoch_loss, epoch_acc

    def evaluate_epoch(self):
        with torch.no_grad():
            self.train_epoch(validate=True)

    def train(self, n_epochs, logs_dir="", val_epoch=1):
        self.bad_epoch = 0
        total_loss, total_acc = 0, 0
        count = 0

        while os.path.exists('runs') and logs_dir in os.listdir('runs'):
            logs_dir = logs_dir.split('_')[0] + f"_{count}"
            count += 1
        self.writer = SummaryWriter(f'runs/{logs_dir}')

        for epoch in range(self.epoch, self.epoch+n_epochs):
            self.epoch = epoch
            loss, acc = self.train_epoch()
            total_loss += loss
            total_acc += acc
            if (epoch+1) % val_epoch == 0:
                self.evaluate_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.bad_epoch == self.patience:
                print("Patience reached.")
                break

        print(f'Total loss {total_loss/n_epochs}. Total acc {total_acc/n_epochs}')
        print('Finished Training')

    def evaluate(self):
        with torch.no_grad():
            return self.train_epoch(validate=True)

    def inference(self, images=None, loader=None):
        target_center = self.target_center
        target_size = self.target_size
        full_img = None
        if images is None:
            if loader is None:
                try:
                    loader = self.valid_loader
                except:
                    print('No Dataloader was specified')
            full_img, in_imgs, target_imgs = next(iter(loader))
            images = in_imgs

        with torch.no_grad():
            top = int(target_center[0] - (target_size / 2))
            left = int(target_center[1] - (target_size / 2))
            self.model.eval()
            images[:, :, top:top + target_size, left:left + target_size] = self.get_model_output(images.float())
            return images, full_img

    def save(self, path, inference=True):
        if inference:
            torch.save(self.model.state_dict(), path)
            print('Model saved. ', path)
        else:
            torch.save({
                'epoch': self.epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'criterion_state_dict': self.criterion.state_dict(),
                'target_center': self.target_center,
                'target_size': self.target_size,
            }, path)
            print('Checkpoint saved. ', path)

    def load(self, path, inference=True):
        if inference:
            self.model.load_state_dict(torch.load(path))
            print('Model loaded. ', path)
        else:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.target_center = checkpoint['target_center']
            self.target_size = checkpoint['target_size']
            print('Checkpoint loaded. ', path)

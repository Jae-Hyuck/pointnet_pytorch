import os
import sys
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('..')  # NOQA
from datasets.indoor3d import Indoor3D
from nets.pointnet_seg import PointNetSeg


class SolverSeg():
    def __init__(self):
        # device cfg
        self.device = torch.device('cuda')

        # Hyper-parameters
        batch_size = 24
        learning_rate = 1e-3

        # Prepare dataset
        train_dataset = Indoor3D(mode='train')
        test_dataset = Indoor3D(mode='test')
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=8)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                      shuffle=False, num_workers=8)

        logging.info(f'# of train shapes: {len(train_dataset)}')
        logging.info(f'# of test shapes:: {len(test_dataset)}')

        # Network
        self.net = PointNetSeg().to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)

        self.identity = torch.eye(64).to(self.device)

    def fit(self):
        num_epochs = 50

        # Start training
        for epoch in range(num_epochs):
            self.net = self.net.train()
            self.scheduler.step()

            total_correct = 0
            total_seen = 0
            total_losses = []
            for i, x in enumerate(self.train_loader):
                # Prepare data
                pts = torch.tensor(x['pts'], dtype=torch.float, device=self.device)
                labels = torch.tensor(x['labels'], dtype=torch.long, device=self.device)

                # Forward pass (pred: Bx13x4096, T2: Bx4096)
                pred, T2 = self.net(pts)

                # Compute loss
                pred_loss = F.cross_entropy(pred, labels)
                reg_loss = F.mse_loss(torch.bmm(T2, T2.transpose(1, 2)),
                                      self.identity.expand(T2.shape[0], -1, -1))
                total_loss = pred_loss + 1.0 * reg_loss

                # Backprop and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Stats
                max_index = torch.max(pred, dim=1)[1]
                correct = (max_index == labels).sum().item()

                total_correct += correct
                total_seen += labels.numel()
                total_losses.append(total_loss.item())

            # Epoch Summary
            mean_loss = sum(total_losses) / len(total_losses)
            mean_accuracy = total_correct / total_seen
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                         f'Loss: {mean_loss:.4f}, '
                         f'Accuracy: {mean_accuracy:.4f}')

            # Eval
            self.evaluate()

    def evaluate(self):
        self.net = self.net.eval()

        total_correct = 0
        total_seen = 0

        with torch.no_grad():
            for i, x in enumerate(self.test_loader):
                # Prepare data
                pts = torch.tensor(x['pts'], dtype=torch.float, device=self.device)
                labels = torch.tensor(x['labels'], dtype=torch.long, device=self.device)

                # Forward pass
                pred, T2 = self.net(pts)

                # Count Accuracy
                max_index = torch.max(pred, dim=1)[1]
                correct = (max_index == labels).sum()

                total_correct += correct.item()
                total_seen += labels.numel()

        # Print Accuracy
        total_accuracy = total_correct / total_seen
        logging.info(f'Eval Accuracy: {total_accuracy:.4f}')

    def save_net(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        filename = 'seg.ckpt'
        torch.save(self.net.state_dict(), os.path.join(save_dir, filename))

    def load_net(self, load_dir):
        filename = 'seg.ckpt'
        self.net.load_state_dict(torch.load(os.path.join(load_dir, filename)))

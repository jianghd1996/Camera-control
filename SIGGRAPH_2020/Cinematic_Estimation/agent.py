import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from tqdm import tqdm
from dataset import estimationdataset
from net import combined_CNN
from collections import OrderedDict
from tensorboardX import SummaryWriter

cudnn.benchmark = True


class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.step = 0

    def tick(self):
        self.step += 1

    def tock(self):
        self.epoch += 1

    def make_ckpt(self):
        return {
            "epoch": self.epoch,
            "step" : self.step
        }

    def load_ckpt(self, ckpt):
        self.epoch = ckpt['epoch']
        self.step = ckpt['step']

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class agent(object):
    def __init__(self,
                 train_data_path,
                 val_data_path,
                 config,
                 ):

        self.config = config
        self.load_data(train_data_path, val_data_path)
        self.clock = TrainClock()
        self.build_model()

        self.train_tb = SummaryWriter(os.path.join(config.log_dir, 'train.events'))
        self.valid_tb = SummaryWriter(os.path.join(config.log_dir, 'valid.events'))

    def load_data(self, train_path, val_path):
        print("loading data")
        train_data = []
        train_label = []
        val_data = []
        val_label = []

        for v in train_path:
            data, label = np.load(os.path.join(v, "toric_estimation_data.npy"), allow_pickle=True)
            train_data.append(data)
            train_label += label

        for v in val_path:
            data, label = np.load(os.path.join(v, "toric_estimation_data.npy"), allow_pickle=True)
            val_data.append(data)
            val_label += label

        self.train_loader = torch.utils.data.DataLoader(
            estimationdataset(train_data, train_label),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True)

        self.val_loader = torch.utils.data.DataLoader(
            estimationdataset(val_data, val_label),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True)

        self.val_loader = cycle(self.val_loader)

    def build_model(self):
        print("building model")

        self.model = combined_CNN(self.config)

        if (self.config.pretrain):
            self.load_ckpt(self.config.load_name)

        self.policies = self.model.parameters()

        self.optimizer = torch.optim.Adam(self.policies, lr=self.config.learning_rate, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=self.config.weight_decay, amsgrad=False)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.reduce_rate)
        self.criterion = torch.nn.MSELoss().cuda()

        self.model = torch.nn.DataParallel(self.model).cuda()

    def learn(self):

        for e in range(self.clock.epoch, self.config.epoch):
            pbar = tqdm(self.train_loader)
            self.model.train()
            for b, data in enumerate(pbar):
                data0, data1, data2, data3, label = data
                data0 = data0.cuda()
                data1 = data1.cuda()
                data2 = data2.cuda()
                data3 = data3.cuda()
                outputs, losses = self.forward([data0, data1, data2, data3], label)
                self.update_network(losses)
                self.clock.tick()
                pbar.set_description("EPOCH[{}][{}]".format(e, b))
                pbar.set_postfix(OrderedDict({"loss": losses.item()}))

                self.record_losses(losses, "train")

                if self.clock.step % self.config.eval_freq == 0:
                    self.model.eval()
                    data = next(self.val_loader)
                    data0, data1, data2, data3, label = data
                    data0 = data0.cuda()
                    data1 = data1.cuda()
                    data2 = data2.cuda()
                    data3 = data3.cuda()
                    with torch.no_grad():
                        outputs, losses = self.forward([data0, data1, data2, data3], label)
                        self.record_losses(losses, "valid")
                    self.model.train()

            self.update_learning_rate()
            self.clock.tock()

            if self.clock.epoch % self.config.save_freq == 0:
                self.save_ckpt(self.clock.epoch)
            self.save_ckpt()

    def forward(self, data, label):
        output = self.model(data)
        loss = self.criterion(output, label.cuda())
        return output, loss

    def record_losses(self, loss, mode='train'):
        if mode == "train":
            self.train_tb.add_scalar("loss", loss, self.clock.step)
        else:
            self.valid_tb.add_scalar("loss", loss, self.clock.step)

    def update_network(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.train_tb.add_scalar("learning_rate", self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        self.scheduler.step(self.clock.epoch)

    def save_ckpt(self, name=None):
        if name is None:
            save_path = os.path.join(self.config.model_path, "latest.pth.tar")
        else:
            save_path = os.path.join(self.config.model_path, "{}.pth.tar".format(name))

        torch.save({
            "model_state_dict": self.model.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "clock_state_dict": self.clock.make_ckpt(),
        }, save_path)

        self.model.cuda()

    def load_ckpt(self, name=None):
        if name is None:
            load_path = os.path.joint(self.config.model_path, "latest.pth.tar")
        else:
            load_path = os.path.join(self.config.model_path, "{}.pth.tar".format(name))

        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.clock.load_ckpt(checkpoint["clock_state_dict"])

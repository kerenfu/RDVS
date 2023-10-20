import argparse
from dataset import Dataset
import torch
from collections import OrderedDict
from model.Depth import Depth
import os
import IOU
import datetime
from torchvision import transforms
import transform
from torch.utils import data

p = OrderedDict()
p['lr_bone'] = 1e-4  # Learning rate
p['lr_branch'] = 1e-3
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [9, 20]
showEvery = 50

CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)


def structure_loss(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask)
    return bce + iou


parser = argparse.ArgumentParser()

# Hyper-parameters11111111111111111111
print(torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda
# pretrain
parser.add_argument('--epoch', type=int, default=25)
parser.add_argument('--epoch_save', type=int, default=5)
parser.add_argument('--save_fold', type=str, default='./checkpoints/depth')  # 训练过程中输出的保存路径
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_thread', type=int, default=0)
parser.add_argument('--model_path', type=str, default='')

# Misc
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

if not os.path.exists(config.save_fold):
    os.mkdir(config.save_fold)

composed_transforms_ts = transforms.Compose([
    transform.RandomHorizontalFlip(),
    transform.FixedResize(size=(config.input_size, config.input_size)),
    transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    transform.ToTensor()])
dataset = Dataset(datasets=['FBMS', 'DAVIS', 'DAVSOD'], transform=composed_transforms_ts, mode='train')
train_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_thread, drop_last=True,
                               shuffle=True)


class Solver(object):
    def __init__(self, train_loader, config, save_fold=None):

        self.train_loader = train_loader
        self.config = config
        self.save_fold = save_fold
        self.build_model()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 返回一个tensor变量内所有元素个数
        print(name)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        print('mode: {}'.format(self.config.mode))
        print('------------------------------------------')
        self.net_bone = Depth(3, mode=self.config.mode)

        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()

        if self.config.mode == 'train':
            if self.config.model_path != '':
                print('load model……')
                assert (os.path.exists(self.config.model_path)), ('please import correct pretrained model path!')
                self.net_bone.load_pretrain_model(self.config.model_path)

        base, head = [], []
        for name, param in self.net_bone.named_parameters():
            if 'depth_bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        self.optimizer_bone = torch.optim.SGD([{'params': base}, {'params': head}], lr=p['lr_bone'],
                                              momentum=p['momentum'], weight_decay=p['wd'], nesterov=True)
        print('------------------------------------------')
        self.print_network(self.net_bone, 'Depth')
        print('------------------------------------------')

    def train(self):
        # 一个epoch中训练iter_num个batch
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        for epoch in range(self.config.epoch):
            self.optimizer_bone.param_groups[0]['lr'] = p['lr_bone']
            self.optimizer_bone.param_groups[1]['lr'] = p['lr_branch']
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                label, depth = data_batch['label'], data_batch['depth']
                if depth.size()[2:] != label.size()[2:]:
                    print("Skip this batch")
                    continue
                if self.config.cuda:
                    label, depth = label.cuda(), depth.cuda()

                decoder_out1_depth, decoder_out2_depth, decoder_out3_depth, decoder_out4_depth, decoder_out5_depth = self.net_bone(depth)

                loss1 = structure_loss(decoder_out1_depth, label)
                loss2 = structure_loss(decoder_out2_depth, label)
                loss3 = structure_loss(decoder_out3_depth, label)
                loss4 = structure_loss(decoder_out4_depth, label)
                loss5 = structure_loss(decoder_out5_depth, label)

                loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

                self.optimizer_bone.zero_grad()
                loss.backward()

                self.optimizer_bone.step()

                if i % showEvery == 0:
                    print(
                        '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f  || sum : %10.4f' % (
                            datetime.datetime.now(), epoch, self.config.epoch, i, iter_num,
                            loss1.data, loss.data))
                    print('Learning rate: ' + str(self.optimizer_bone.param_groups[0]['lr']))

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           '%s/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if epoch in lr_decay_epoch:
                p['lr_bone'] = p['lr_bone'] * 0.2
                p['lr_branch'] = p['lr_branch'] * 0.2

        torch.save(self.net_bone.state_dict(), '%s/final_bone.pth' % self.config.save_fold)


train = Solver(train_loader, config)
train.train()

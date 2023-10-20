import argparse
import os
from dataset import Dataset
import torch
from torchvision import transforms
import transform
from torch.utils import data
from model.DCTNet import Model
import numpy as np
import cv2

parser = argparse.ArgumentParser()
print(torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

# test
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--num_thread', type=int, default=0)
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--spatial_ckpt', type=str, default=None)
parser.add_argument('--flow_ckpt', type=str, default=None)
parser.add_argument('--depth_ckpt', type=str, default=None)
parser.add_argument('--model_path', type=str, default='./checkpoints/DCTNet/final_bone.pth')
parser.add_argument('--test_dataset', type=list, default=['DAVIS', 'FBMS', 'SegTrack-V2', 'DAVSOD', 'VOS'])
# parser.add_argument('--test_dataset', type=list, default=['RDVS-TEST'])
# parser.add_argument('--test_dataset', type=list, default=['RDVS-SYNTHETIC'])
parser.add_argument('--testsavefold', type=str, default='./DCTNet_plus')

# Misc
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
config = parser.parse_args()

composed_transforms_te = transforms.Compose([
    transform.FixedResize(size=(config.input_size, config.input_size)),
    transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    transform.ToTensor()])

dataset = Dataset(datasets=config.test_dataset, transform=composed_transforms_te, mode='test')
test_loader = data.DataLoader(dataset, batch_size=config.test_batch_size, num_workers=config.num_thread,
                              drop_last=True, shuffle=False)

print('mode: {}'.format(config.mode))
print('------------------------------------------')
net_bone = Model(3, mode=config.mode, spatial_ckpt=config.spatial_ckpt,
                 flow_ckpt=config.flow_ckpt, depth_ckpt=config.depth_ckpt)
# net_bone = Model(3, mode=config.mode, model_path=None)
name = "DCTNet"
if config.cuda:
    net_bone = net_bone.cuda()
assert (config.model_path != ''), ('Test mode, please import pretrained model path!')
assert (os.path.exists(config.model_path)), ('please import correct pretrained model path!')
print('load model……all checkpoints')
# ckpt = torch.load(config.model_path)
# model_dict = net_bone.state_dict()
# pretrained_dict = {k[7:]: v for k, v in ckpt.items() if k[7:] in model_dict}
# model_dict.update(pretrained_dict)
# net_bone.load_state_dict(model_dict)
net_bone.load_pretrain_model(config.model_path)
net_bone.eval()

if not os.path.exists(config.testsavefold):
    os.makedirs(config.testsavefold)

for i, data_batch in enumerate(test_loader):
    print("progress {}/{}\n".format(i + 1, len(test_loader)))
    image, flow, depth, name, split, size = data_batch['image'], data_batch['flow'], data_batch['depth'], \
                                            data_batch['name'], data_batch['split'], data_batch['size']
    dataset = data_batch['dataset']

    if config.cuda:
        image, flow, depth = image.cuda(), flow.cuda(), depth.cuda()
    with torch.no_grad():

        decoder_out1, decoder_out2, decoder_out3, decoder_out4, decoder_out5, sc_out = net_bone(
            image, flow, depth)

        for i in range(config.test_batch_size):
            presavefold = os.path.join(config.testsavefold, dataset[i], split[i])

            if not os.path.exists(presavefold):
                os.makedirs(presavefold)
            pre1 = torch.nn.Sigmoid()(decoder_out1[i])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
            pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
            pre1 = cv2.resize(pre1, (int(size[0][1]), int(size[0][0])))
            cv2.imwrite(presavefold + '/' + name[i], pre1)

            # save Coarse Map
            # sc = torch.nn.Sigmoid()(sc_out[i])
            # sc = (sc - torch.min(sc)) / (torch.max(sc) - torch.min(sc))
            # sc = np.squeeze(sc.cpu().data.numpy()) * 255
            # sc = cv2.resize(sc, (int(size[0][1]), int(size[0][0])))
            # cv2.imwrite(presavefold + '/' + name[i], sc)
           

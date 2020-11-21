from datetime import datetime
import os
import os.path as osp

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer

from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from tqdm import tqdm

local_path = osp.dirname(osp.abspath(__file__))


def centroids_init(model, data_dir, datasetTrain, composed_transforms):
    centroids = torch.zeros(3, 304, 64, 64).cuda() # 3 means the number of source domains
    model.eval()

    # Calculate initial centroids only on training data.
    with torch.set_grad_enabled(False):
        count = 0
        # tranverse each training source domain
        for index in datasetTrain:
            domain = DL.FundusSegmentation(base_dir=data_dir, phase='train', splitid=[index],
                                           transform=composed_transforms)
            dataloder = DataLoader(domain, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

            for id, sample in tqdm(enumerate(dataloder)):
                sample=sample[0]
                inputs = sample['image'].cuda()
                features = model(inputs, extract_feature=True)

                # Calculate the sum features from the same domain
                centroids[count:count+1] += features

            # Average summed features with class count
            centroids[count] /= torch.tensor(len(dataloder)).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
            count += 1
    # Calculate the mean features for each domain
    ave = torch.mean(torch.mean(centroids, 3, True), 2, True) # size [3, 304]
    return ave.expand_as(centroids).contiguous()  # size [3, 304, 64, 64]

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')

    parser.add_argument('--datasetTrain', nargs='+', type=int, default=1, help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=1, help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max-epoch', type=int, default=120, help='max epoch')
    parser.add_argument('--stop-epoch', type=int, default=80, help='stop epoch')
    parser.add_argument('--interval-validate', type=int, default=10, help='interval epoch number to valide the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
    parser.add_argument('--lam', type=float, default=0.9, help='momentum of memory update',)
    parser.add_argument('--data-dir', default='../../../../Dataset/Fundus/', help='data root path')
    parser.add_argument('--pretrained-model', default='../../../models/pytorch/fcn16s_from_caffe.pth', help='pretrained model of FCN16s',)
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+',)
    args = parser.parse_args()

    now = datetime.now()
    args.out = osp.join(local_path, 'logs', 'test'+str(args.datasetTest[0]), 'lam'+str(args.lam), now.strftime('%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    torch.cuda.manual_seed(1337)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(256),
        # tr.RandomCrop(512),
        # tr.RandomRotate(),
        # tr.RandomFlip(),
        # tr.elastic_transform(),
        # tr.add_salt_pepper_noise(),
        # tr.adjust_light(),
        # tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='train', splitid=args.datasetTrain,
                                                         transform=composed_transforms_tr)
    train_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                       transform=composed_transforms_ts)
    val_loader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. model
    model = DeepLab(num_classes=2, num_domain=3, backbone='mobilenet', output_stride=args.out_stride, lam=args.lam).cuda()
    print('parameter numer:', sum([p.numel() for p in model.parameters()]))

    # load weights
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        print('Before ', model.centroids.data)
        model.centroids.data = centroids_init(model, args.data_dir, args.datasetTrain, composed_transforms_ts)
        print('Before ', model.centroids.data)
        # model.freeze_para()

    start_epoch = 0
    start_iteration = 0

    # 3. optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99)
    )

    trainer = Trainer.Trainer(
        cuda=cuda,
        model=model,
        lr=args.lr,
        lr_decrease_rate=args.lr_decrease_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()

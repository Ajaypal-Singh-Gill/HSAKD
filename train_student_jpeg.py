import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np


import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds


from bisect import bisect_right
import time
import math
from JPEG_layer import *


# Define the custom module
class CustomModel(nn.Module):
    def __init__(self, jpeg_layer, underlying_model):
        super(CustomModel, self).__init__()
        self.jpeg_layer = jpeg_layer
        self.underlying_model = underlying_model

    def forward(self, x, *args, **kwargs):
        # Pass input through jpeg_layer
        x = self.jpeg_layer(x)
        # Pass the output of jpeg_layer to the underlying_model
        # along with any additional arguments
        self.underlying_model.eval()
        x = self.underlying_model(x, *args, **kwargs)
        return x

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='wrn_16_2_aux', type=str, help='student network architecture')
parser.add_argument('--tarch', default='wrn_40_2_aux', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='wrn_40_2_aux.pth.tar', type=str, help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150,180,210], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=300, type=int, dest='sgdr_t',help='SGDR T_0')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=240, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')

# JPEG layer is added
parser.add_argument('--lr_decay_epochs', type=str, default=[150,180,210], help='where to decay lr, can be a list')

parser.add_argument('--JPEG_enable', action='store_true')
parser.add_argument('--JPEG_alpha_trainable', action='store_true')
parser.add_argument('--JPEG_alpha', type=float, default=10.0, help='Tempurature scaling')
parser.add_argument('--JPEG_learning_rate', type=float, default=0.0125, help='Quantization Table Learning Rate')
parser.add_argument('--alpha_learning_rate', type=float, default=None, help='Alpha Learning Rate')
parser.add_argument('--Q_inital', type=float, default=1.0, help='Initial Quantization Step')
parser.add_argument('--block_size', type=int, default=8, help='the experiment id')
parser.add_argument('--num_bit', type=int, default=8, help='Number of bits to represent DCT coeff')
parser.add_argument('--min_Q_Step', type=float, default=1.0, help='Minumum Quantization Step')
parser.add_argument('--max_Q_Step', type=float, default=255, help='Maximum Quantization Step')
parser.add_argument('--num_non_zero_q', type=int, default=5, choices=range(2,2**10 - 1), help='Window size for the reconstruction space')
parser.add_argument('--log_file', type=str, default='_alpha_untrainable', help='add text to the file', choices=['', '_alpha_per_q', '_alpha_per_channel', '_alpha_untrainable'])
parser.add_argument('--process_id', type=int, default=1, help='process id')


# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

log_txt = 'result/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'tarch' + '_' +  args.tarch + '_'+\
          'arch' + '_' +  args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed) +'.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'tarch' + '_' +  args.tarch + '_'+\
          'arch'+ '_' + args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed)

args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if args.resume is False and args.evaluate is False:
    with open(log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.set_printoptions(precision=4)

# if opt.dataset == 'cifar100' or opt.dataset == 'cifar10':
mean=(0.5071, 0.4867, 0.4408)
std=(0.2675, 0.2565, 0.2761)
mean_datatloader=(0, 0, 0)
std_datatloader=(1/255., 1/255., 1/255.)

# else:
#     raise NotImplementedError(opt.dataset)


num_classes = 100
trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                [0.2675, 0.2565, 0.2761])
                                        ]))

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                [0.2675, 0.2565, 0.2761]),
                                        ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=(torch.cuda.is_available()))

# For JPEG Layer
trainset_jpeg = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0, 0, 0],
                                                                [1/255., 1/255., 1/255.])
                                        ]))

testset_jpeg = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0, 0, 0],
                                                                [1/255., 1/255., 1/255.]),
                                        ]))

trainloader_jpeg = torch.utils.data.DataLoader(trainset_jpeg, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=(torch.cuda.is_available()))

testloader_jpeg = torch.utils.data.DataLoader(testset_jpeg, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=(torch.cuda.is_available()))


print('==> Building model..')
net = getattr(models, args.tarch)(num_classes=num_classes)
net.eval()
resolution = (1, 3, 32, 32)
print('Teacher Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.tarch, cal_param_size(net)/1e6, cal_multi_adds(net, resolution)/1e9))
del(net)
net = getattr(models, args.arch)(num_classes=num_classes)
net.eval()
resolution = (1, 3, 32, 32)
print('Student Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.arch, cal_param_size(net)/1e6, cal_multi_adds(net, resolution)/1e9))
del(net)


print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))     
checkpoint = torch.load(args.tcheckpoint, map_location=torch.device('cpu'))

model = getattr(models, args.arch)
net = model(num_classes=num_classes).cuda()
net =  torch.nn.DataParallel(net)

tmodel = getattr(models, args.tarch)
tnet = tmodel(num_classes=num_classes).cuda()
tnet.load_state_dict(checkpoint['net'])
# tnet.eval()
tnet =  torch.nn.DataParallel(tnet)

_, ss_logits = net(torch.randn(2, 3, 32, 32))
num_auxiliary_branches = len(ss_logits)
cudnn.benchmark = True

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch*all_iters_per_epoch)/(args.warmup_epoch *all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)
    
    print(f"Type of lr_decay_epochs: {type(args.lr_decay_epochs)}", args.lr_decay_epochs)

    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    if steps > 0:
        new_lr_JPEG = args.JPEG_learning_rate * (args.lr_decay_rate ** steps)
        param_group[0]['lr'] = cur_lr
        param_group[1]['lr'] = new_lr_JPEG
        param_group[2]['lr'] = new_lr_JPEG
        if args.JPEG_alpha_trainable: 
            alpha_learning_rate = args.alpha_learning_rate * (args.lr_decay_rate ** steps)
            param_group[2]['lr'] = alpha_learning_rate

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = cur_lr
    return cur_lr


def train(epoch, criterion_list, optimizer):
    train_loss = 0.
    train_loss_cls = 0.
    train_loss_div = 0.

    ss_top1_num = [0] * num_auxiliary_branches
    ss_top5_num = [0] * num_auxiliary_branches
    class_top1_num = [0] * num_auxiliary_branches
    class_top5_num = [0] * num_auxiliary_branches
    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    net.train()
    mean=(0.5071, 0.4867, 0.4408)
    std=(0.2675, 0.2565, 0.2761)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.]),
        transforms.Normalize(mean, std),
    ])
    for batch_idx, (input, target) in enumerate(trainloader_jpeg): # JPEG Change
        batch_start_time = time.time()
        input = input.float().cuda()
        target = target.cuda()

        size = input.shape[1:]
        input = torch.stack([torch.rot90(input, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
        labels = torch.stack([target*4+i for i in range(4)], 1).view(-1)

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        optimizer.zero_grad()
        logits, ss_logits = net(transform(input), grad=True)
        # print("Parameters of CustomModel with JPEG Layer TRAIN:")
        # for name, param in tnet.named_parameters():
        #     print(name, param.size(), param.requires_grad)
        # break

        t_logits, t_ss_logits = tnet(input)
        t_logits_detached = t_logits.detach()
        t_ss_logits = [ss_logit.detach() for ss_logit in t_ss_logits]        
        # with torch.no_grad():
        #     t_logits, t_ss_logits = tnet(input)

        loss_cls = torch.tensor(0.).cuda()
        loss_div = torch.tensor(0.).cuda()

        loss_cls = loss_cls + criterion_cls(logits[0::4], target)
        for i in range(len(ss_logits)):
            loss_div = loss_div + criterion_div(ss_logits[i], t_ss_logits[i])
        
        loss_div = loss_div + criterion_div(logits, t_logits)
        
            
        loss = loss_cls + loss_div
        loss.backward()
        optimizer.step()
        # for name, param in jpeg_layer.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)


        train_loss += loss.item() / len(trainloader)
        train_loss_cls += loss_cls.item() / len(trainloader)
        train_loss_div += loss_div.item() / len(trainloader)

        for i in range(len(ss_logits)):
            top1, top5 = correct_num(ss_logits[i], labels, topk=(1, 5))
            ss_top1_num[i] += top1
            ss_top5_num[i] += top5
        
        class_logits = [torch.stack(torch.split(ss_logits[i], split_size_or_sections=4, dim=1), dim=1).sum(dim=2) for i in range(len(ss_logits))]
        multi_target = target.view(-1, 1).repeat(1, 4).view(-1)
        for i in range(len(class_logits)):
            top1, top5 = correct_num(class_logits[i], multi_target, topk=(1, 5))
            class_top1_num[i] += top1
            class_top5_num[i] += top5

        logits = logits.view(-1, 4, num_classes)[:, 0, :]
        top1, top5 = correct_num(logits, target, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += target.size(0)

        # print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
        #     epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (top1_num/(total)).item()))


    ss_acc1 = [round((ss_top1_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)]
    ss_acc5 = [round((ss_top5_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)]
    class_acc1 = [round((class_top1_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)] + [round((top1_num/(total)).item(), 4)]
    class_acc5 = [round((class_top5_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)] + [round((top5_num/(total)).item(), 4)]
    
    print('Train epoch:{}\nTrain Top-1 ss_accuracy: {}\nTrain Top-1 class_accuracy: {}\n'.format(epoch, str(ss_acc1), str(class_acc1)))

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                '\n train_loss:{:.5f}\t train_loss_cls:{:.5f}\t train_loss_div:{:.5f}'
                '\nTrain Top-1 ss_accuracy: {}\nTrain Top-1 class_accuracy: {}\n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls, train_loss_div,
                        str(ss_acc1), str(class_acc1)))



def test(epoch, criterion_cls, net):
    global best_acc
    test_loss_cls = 0.

    ss_top1_num = [0] * (num_auxiliary_branches)
    ss_top5_num = [0] * (num_auxiliary_branches)
    class_top1_num = [0] * num_auxiliary_branches
    class_top5_num = [0] * num_auxiliary_branches
    top1_num = 0
    top5_num = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            batch_start_time = time.time()
            input, target = inputs.cuda(), target.cuda()

            size = input.shape[1:]
            input = torch.stack([torch.rot90(input, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
            labels = torch.stack([target*4+i for i in range(4)], 1).view(-1)
            
            logits, ss_logits = net(input)
            loss_cls = torch.tensor(0.).cuda()
            loss_cls = loss_cls + criterion_cls(logits[0::4], target)

            test_loss_cls += loss_cls.item()/ len(testloader)

            batch_size = logits.size(0) // 4
            for i in range(len(ss_logits)):
                top1, top5 = correct_num(ss_logits[i], labels, topk=(1, 5))
                ss_top1_num[i] += top1
                ss_top5_num[i] += top5
                
            class_logits = [torch.stack(torch.split(ss_logits[i], split_size_or_sections=4, dim=1), dim=1).sum(dim=2) for i in range(len(ss_logits))]
            multi_target = target.view(-1, 1).repeat(1, 4).view(-1)
            for i in range(len(class_logits)):
                top1, top5 = correct_num(class_logits[i], multi_target, topk=(1, 5))
                class_top1_num[i] += top1
                class_top5_num[i] += top5

            logits = logits.view(-1, 4, num_classes)[:, 0, :]
            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)
            

            # print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
            #     epoch, batch_idx, len(testloader), time.time()-batch_start_time, (top1_num/(total)).item()))

        ss_acc1 = [round((ss_top1_num[i]/(total*4)).item(), 4) for i in range(len(ss_logits))]
        ss_acc5 = [round((ss_top5_num[i]/(total*4)).item(), 4) for i in range(len(ss_logits))]
        class_acc1 = [round((class_top1_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)] + [round((top1_num/(total)).item(), 4)]
        class_acc5 = [round((class_top5_num[i]/(total*4)).item(), 4) for i in range(num_auxiliary_branches)] + [round((top5_num/(total)).item(), 4)]
        with open(log_txt, 'a+') as f:
            f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 ss_accuracy: {}\nTop-1 class_accuracy: {}\n'
                    .format(epoch, test_loss_cls, str(ss_acc1), str(class_acc1)))
        print('test epoch:{}\nTest Top-1 ss_accuracy: {}\nTest Top-1 class_accuracy: {}\n'.format(epoch, str(ss_acc1), str(class_acc1)))

    return class_acc1[-1]


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)

    if args.evaluate: 
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))     
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        test(start_epoch, criterion_cls, net)
    else:
        print('Evaluate Teacher:')
        acc = test(0, criterion_cls, tnet)
        print('Teacher Acc:', acc)


        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        
        jpeg_layer = JPEG_layer(opt=args, img_shape=trainloader_jpeg.dataset.data[0].shape, mean=mean, std=std)

        optimizer_data = [
                    {'params': trainable_list.parameters(), 'lr': 0.1, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': True}
                ]
        optimizer_data.append({'params': [jpeg_layer.lum_qtable], 'lr': args.JPEG_learning_rate, 'momentum': 0.9})
        optimizer_data.append({'params': [jpeg_layer.chrom_qtable], 'lr': (args.JPEG_learning_rate)/np.sqrt(2), 'momentum': 0.9})
        if args.JPEG_alpha_trainable:
            optimizer_data.append({'params': jpeg_layer.alpha_lum, 'lr': args.alpha_learning_rate, 'momentum': 0.9})
            optimizer_data.append({'params': jpeg_layer.alpha_chrom, 'lr': (args.alpha_learning_rate)/np.sqrt(2), 'momentum': 0.9})

        optimizer = optim.SGD(optimizer_data)

        tnet = CustomModel(jpeg_layer, tnet)
        # print("Parameters of CustomModel with JPEG Layer:")
        # for name, param in tnet.named_parameters():
        #     print(name, param.size(), param.requires_grad)
        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.cuda()


        if args.resume:
            print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
        
        # print("JPEG Layer parameters before training:")
        # for name, param in jpeg_layer.named_parameters():
        #     print(name, param.requires_grad)

        for epoch in range(start_epoch, args.epochs):

            with torch.no_grad(): 
                lum_qtable =  tnet.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                chrom_qtable =  tnet.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0)
                # print(quantizationTable)
                print("Quantization Table --> Min: {:.2f}, Max: {:.2f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
            
            if args.JPEG_alpha_trainable:
                with torch.no_grad(): 
                    alpha_lum =  tnet.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                    alpha_chrom = tnet.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                    alpha = torch.cat((alpha_lum, alpha_chrom), 0)
                    print("Alpha --> Min: {:.5f}, Max: {:.5f}".format(alpha.min().item(), alpha.max().item()))


            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls, net)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls, net)

        with open(log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + log_txt + ' ' + args.checkpoint_dir)

import argparse
import os
import shutil
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

torch.manual_seed(1)

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)
print(torch.cuda.device_count())

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


#https://stackoverflow.com/questions/38340311/what-is-the-difference-between-steps-and-epochs-in-tensorflow
#Number of training steps per epoch: total_number_of_training_examples / batch_size.
#Total number of training steps: number_of_epochs x Number of training steps per epoch.
#total of 170 2/3 epochs.
parser.add_argument('--epochs', default=171, type=int, metavar='N',
                    help='number of total epochs to run')


parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')

parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')


parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default= 0.0002, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

#per 391 since tf uses per 391 steps 
parser.add_argument('--print-freq', '-p', default=390, type=int,
                    metavar='N', help='print frequency (default: 50)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')

parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)

parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

#############################
parser.add_argument('--gpuid', '--gpuid', default= 3, type=int,
                    metavar='gpuid', help='gpuid')

parser.add_argument('--miu', '--miu', default= 0, type=float,
                    metavar='miu', help='miu for lamba equivalent')

parser.add_argument('--lamb', '--lamb', default= 0, type=int,
                    metavar='lamb', help='lambda')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #sets to gpuid
    torch.cuda.set_device(args.gpuid)
    model = torch.nn.DataParallel(resnet.resnet32(), device_ids=[args.gpuid,])
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=250, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    if args.miu:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum)

    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


    # if args.lamb > 0:
    #     for g in optimizer.param_groups:
    #         g['lr'] = 1 - g['lr']


    #multistepLR has a default decay rate of 0.1 for the LEARNING RATE (not for weights), added the decay per 10k steps, per 21~ epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[22,43,64,85,101,128,150,],
                                                        last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    total_train_loss = []
    total_train_error = []
    total_train_regloss = []
    total_vali_error = []

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_loss, train_error, train_regloss = train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        vali_error, prec1 = validate(val_loader, model, criterion)

        ##print out the loss/error you get and csv it. HERE.
        total_vali_error.extend(vali_error)
        total_train_loss.extend(train_loss)
        total_train_error.extend(train_error)
        total_train_regloss.extend(train_regloss)

        print(f'total_train_loss: {total_train_loss}')
        print(f'total_train_error: {total_train_error}')
        print(f'total_train_regloss: {total_train_regloss}')
        print(f'total_vali_error: {total_vali_error}')

        #tocsv(all those losses/errors)
        df = pd.DataFrame(data= {'train_error': total_train_error, 'vali_error': total_vali_error, \
                            'train_loss':total_train_loss, 'train_regloss': total_train_regloss})

        df.to_csv(os.path.join(args.save_dir, 'model_error.csv'))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_error = AverageMeter()
    reg_loss = AverageMeter()

    train_loss = []
    train_error = []
    train_regloss = []

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        print('*'*20 + 'Starting new batch '+ str(i) + '*'*20)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)

        #criterion is nn.crossEntropy()
        loss = criterion(output, target_var)
        print(f'lost w/o reg: {loss.item()}')

        #Autograd then calculates and stores the gradients for each model parameter in the parameter’s .grad attribute.
        #create our own weight update method
        #https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/3  
        #Using CG by updating weights through eqn7

        if args.lamb > 0:
            print('*'*10 + "USING CG" + '*'*10)
            model.zero_grad()
            loss.backward()

            frob_reg_loss = 0

            with torch.no_grad():
                for param in model.parameters():
                    #to track the value of R(.) which is the L2norm/ Frobnorm
                    frob_reg_loss += (torch.norm(param, p='fro') + 0.0000000001)**1 #square it to if you want to make it L2norm

                    #does the calc of St and change the gradient of each parameter to factor St in.
                    frob_grad = torch.norm(param.grad, p = 'fro') + 0.0000000001 #equivalent to cgd_fn
                    param.grad = param + (args.lamb * param.grad/ frob_grad)

            #collect all reg_loss for all batches
            reg_loss.update(frob_reg_loss.item(), input.size(0))

            optimizer.step()
        
        #eqn 2 instead of eqn7 using miu*R(w)            
        elif args.miu > 0:
            print('*'*10 + "USING MIU" + '*'*10)
            #frob norm of weights
            weight_reg = 0

            #https://discuss.pytorch.org/t/how-to-print-models-parameters-with-its-name-and-requires-grad-value/10778/5
            #https://stackoverflow.com/questions/65876372/pytorch-using-param-grad-affects-learning-process-solved
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        #R(.) calculates the frobenius norm with respect to the weights, eqn 2
                        weight_reg += (torch.norm(param, p = 'fro')) + 0.0000000001
                        #https://pytorch.org/docs/stable/generated/torch.norm.html

            #adding the frob norm into loss
            loss = loss + (args.miu * weight_reg)
            print(f'lost with reg: {loss.item()}')

            optimizer.zero_grad()

            #uses the new added loss to update weights but the loss being printed out is solely just cross entropy loss
            loss.backward()
            optimizer.step()

        else:
            # compute gradient and do SGD step
            optimizer.zero_grad()

            # loss.backward() calculates the gradient of weight wrt loss for each weight and stores the gradient value in param.grad
            loss.backward()

            # optimizer.step() performs the update
            optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        print(f'train prec1: {prec1}')

        #measure how many they got it wrong
        error1 = predict_error(output.data, target)[0]

        #update and average out after every batch
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top1_error.update(error1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:#only every 391 batches

            if args.lamb > 0 :
                print(f'frob_reg_loss: {reg_loss.val}')
                train_regloss.append(reg_loss.val)

            #can consider changing .val to .avg is we want the average of entire epoch instead of batches
            print(f'losses : {losses.val}')
            print(f'top1_error: {top1_error.val}')

            #append
            train_loss.append(losses.val)
            train_error.append(top1_error.val)

            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #           epoch, i, len(train_loader), batch_time=batch_time,
            #           data_time=data_time, loss=losses, top1=top1))

    return train_loss, train_error, train_regloss

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_error = AverageMeter()

    vali_error = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            print('*'*10 + 'vali batch no:' + str(i) + '*' *10)
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            # already converted to %. Eg. 9.35
            prec1 = accuracy(output.data, target)[0]

            error1 = predict_error(output.data, target)[0]

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top1_error.update(error1.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 39 == 0: #total_vali_batches = 250
                #can consider changing .val to .avg is we want the average of entire epoch instead of batches
                print(f'losses : {losses.val}')
                vali_error.append(top1_error.val)

                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    # return top1.avg
    return vali_error, top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #pred is the index of the prediction with highest probability score
    _, pred = output.topk(maxk, 1, True, True)

    #pred.t() gets the predicted class in terms of index

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #target.view().expand(pred) means view the target and make it same shape as pred.t()
    #pred.eq() => eq does matching. Eg. [1,1].eq([1,2]) => [True, False]

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        to_append = correct_k.mul_(100.0 / batch_size)
        res.append(to_append)
    return res

def predict_error(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    #pred is the index of the prediction with highest probability score
    _, pred = output.topk(maxk, 1, True, True)

    #pred.t() gets the predicted class in terms of index

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #target.view().expand(pred) means view the target and make it same shape as pred.t()
    #pred.eq() => eq does matching. Eg. [1,1].eq([1,2]) => [True, False]

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) #already a float value
        to_append = 100.0 - correct_k.mul_(100.0 / batch_size)
        print(f'error result: {to_append}')

        res.append(to_append)

    return res

if __name__ == '__main__':
    main()

                                        



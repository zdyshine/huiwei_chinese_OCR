from __future__ import print_function
import argparse
import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
# import models.crnn as crnn
# import models.vgg19_crnn_attention as crnn
import models.crnn_attention as crnn
import re
import params
import codecs


parser = argparse.ArgumentParser()
# parser.add_argument('--trainroot', required=True, help='path to dataset')
# parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

opt = parser.parse_args()
print(opt)

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)
     #___________________________
        preds = crnn(image)
        # preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        list_1 = []
        for i in cpu_texts:
            list_1.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, list_1):
            if pred == target:
                n_correct += 1

    
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    # print(n_correct)
    # print(max_iter*params.batchSize)
    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    with codecs.open('logs/log.txt','a') as f:
        f.write('Test loss: ' + str(loss_avg.val()) + 'accuary:' + str(accuracy) + '\n')


def trainBatch(net, criterion, optimizer, train_iter):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def training():
    for total_steps in range(params.niter):
        if total_steps < 2:
            train_loader = s_train_loader
            test_dataset = s_test_dataset
        elif total_steps > 4:
            train_loader = m_train_loader
            test_dataset = m_test_dataset
        else:
            train_loader = l_train_loader
            test_dataset = l_test_dataset

        train_iter = iter(train_loader)
        i = 0

        print(len(train_loader))
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()
            cost = trainBatch(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1
            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (total_steps, params.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()
            if i % params.valInterval == 0:
                val(crnn, test_dataset, criterion)
        if (total_steps+1) % params.saveInterval == 0:
            torch.save(crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(params.experiment, total_steps, i))

if __name__ == '__main__':

    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    
    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')

    # read s_train set
    s_train_dataset = dataset.lmdbDataset(root=params.s_train_data)
    assert s_train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(s_train_dataset, params.batchSize)
    else:
        sampler = None

    # images will be resize to 32*96
    s_train_loader = torch.utils.data.DataLoader(
        s_train_dataset, batch_size=params.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=96, keep_ratio=params.keep_ratio))

    # read s_train set
    m_train_dataset = dataset.lmdbDataset(root=params.m_train_data)
    assert m_train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(m_train_dataset, params.batchSize)
    else:
        sampler = None

    # images will be resize to 32*384
    m_train_loader = torch.utils.data.DataLoader(
        m_train_dataset, batch_size=params.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=384, keep_ratio=params.keep_ratio))


    # read l_train set
    l_train_dataset = dataset.lmdbDataset(root=params.l_train_data)
    assert l_train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(l_train_dataset, params.batchSize)
    else:
        sampler = None

    # images will be resize to 32*768
    l_train_loader = torch.utils.data.DataLoader(
        l_train_dataset, batch_size=params.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=768, keep_ratio=params.keep_ratio))



    # read test set
    # images will be resize to 32*160
    s_test_dataset = dataset.lmdbDataset(
        root=params.s_test_data, transform=dataset.resizeNormalize((320, 32)))

    m_test_dataset = dataset.lmdbDataset(
        root=params.m_test_data, transform=dataset.resizeNormalize((512, 32)))

    l_test_dataset = dataset.lmdbDataset(
        root=params.l_test_data, transform=dataset.resizeNormalize((768, 32)))

    nclass = len(params.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(params.alphabet)
    # criterion = CTCLoss()
    criterion = nn.CTCLoss(reduction = 'mean')

    # cnn and rnn
    image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
    text = torch.IntTensor(params.batchSize * 5)
    length = torch.IntTensor(params.batchSize)

    crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh)
    print(crnn)
    # print(crnn.parameters())

    # os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu_id
    #
    # cudnn.benchmark = True
    # if torch.cuda.is_available() and params.using_cuda:
    #     cuda = True
    #     print('Using cuda')
    # else:
    #     cuda = False
    #     print('Using cpu mode')

    if opt.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    crnn.apply(weights_init)
    if params.crnn != '':
        print('loading pretrained model from %s' % params.crnn)
        crnn.load_state_dict(torch.load(params.crnn))

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
                               betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)
    elif params.sgd:
        optimizer = optim.SGD(crnn.parameters(), lr=params.lr, momentum=params.momentum)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    training()

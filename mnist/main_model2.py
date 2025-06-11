import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv

from src.time import time
from src.model2 import Model2  # 修改1：導入Model2（假設類名為Model）
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from src.argument import parser, print_args

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger
        opt = torch.optim.Adam(model.parameters(), args.learning_rate)
        _iter = 0
        begin_time = time()

        for epoch in range(1, args.max_epoch+1):
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:
                    with torch.no_grad():
                        std_output = model(data, _eval=True)
                    pred = torch.max(std_output, dim=1)[1]
                    std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    logger.info('epoch: %d, iter: %d, spent %.2f s' % (
                        epoch, _iter, time() - begin_time))
                    logger.info('std_acc: %.3f%%' % (std_acc))

                    if va_loader is not None:
                        va_acc, _ = self.test(model, va_loader, False)
                        logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                        logger.info('test_acc: %.3f%%' % (va_acc*100))
                        logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')
                    begin_time = time()
                _iter += 1

    def test(self, model, loader, adv_test=False):
        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                output = model(data, _eval=True)
                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                total_acc += te_acc
                num += output.shape[0]

        return total_acc / num, total_adv_acc / num

def main(args):
    save_folder = '%s_%s' % (args.dataset, args.affix)
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')
    print_args(args, logger)

    model = Model(i_c=1, n_c=10)  # 修改2：使用Model2初始化
    attack = FastGradientSignUntargeted(model,  # 修改3：攻擊對象改為Model2
                                      args.epsilon, 
                                      args.alpha, 
                                      min_val=0, 
                                      max_val=1, 
                                      max_iters=args.k, 
                                      _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        tr_dataset = tv.datasets.MNIST(args.data_root, 
                                     train=True, 
                                     transform=tv.transforms.ToTensor(), 
                                     download=True)
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        te_dataset = tv.datasets.MNIST(args.data_root, 
                                     train=False, 
                                     transform=tv.transforms.ToTensor(), 
                                     download=True)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        pass
    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)

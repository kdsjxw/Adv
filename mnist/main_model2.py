import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv

from time import time
from src.model.model2 import Model2     # 修改1：導入Model2和對應攻擊模塊
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
        opt = torch.optim.Adam(model.parameters(), args.learning_rate)
        _iter = 0
        begin_time = time()

        for epoch in range(1, args.max_epoch+1):
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    # 修改2：保持與main.py相同的對抗訓練邏輯
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:
                    self._eval_and_log(model, data, label, epoch, _iter, begin_time, va_loader)
                    begin_time = time()
                _iter += 1

    def _eval_and_log(self, model, data, label, epoch, _iter, begin_time, va_loader):
        """封裝評估與日誌記錄（與main.py保持相同格式）"""
        with torch.no_grad():
            std_output = model(data, _eval=True)
        pred = torch.max(std_output, dim=1)[1]
        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

        self.logger.info('epoch: %d, iter: %d, spent %.2f s' % (
            epoch, _iter, time() - begin_time))
        self.logger.info('std_acc: %.3f%%' % (std_acc))

        if va_loader is not None:
            va_acc, va_adv_acc = self.test(model, va_loader, True)  # 修改3：始終測試魯棒性
            self.logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
            self.logger.info('test_acc: %.3f%%' % (va_acc * 100))
            self.logger.info('robust_acc: %.3f%%' % (va_adv_acc * 100))  # 新增魯棒準確率
            self.logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

    def test(self, model, loader, adv_test=False):
        """修改4：完整實現魯棒性測試（與main.py相同邏輯）"""
        total_acc = 0.0
        total_adv_acc = 0.0
        num = 0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                
                # 標準測試
                output = model(data, _eval=True)
                pred = torch.max(output, dim=1)[1]
                total_acc += evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                # 對抗測試
                if adv_test:
                    adv_data = self.attack.perturb(data, label, 'mean', False)
                    adv_output = model(adv_data, _eval=True)
                    adv_pred = torch.max(adv_output, dim=1)[1]
                    total_adv_acc += evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                num += output.shape[0]

        return total_acc / num, total_adv_acc / num

def main(args):
    # 初始化路徑（與main.py完全一致）
    save_folder = '%s_%s' % (args.dataset, args.affix)
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)
    makedirs(log_folder)
    makedirs(model_folder)
    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    # 修改5：使用Model2但保持相同初始化接口
    logger = create_logger(log_folder, args.todo, 'info')
    print_args(args, logger)
    model = Model2(i_c=1, n_c=10)  # MNIST默認參數
    attack = FastGradientSignUntargeted(model, 
                                      args.epsilon, 
                                      args.alpha, 
                                      min_val=0, 
                                      max_val=1, 
                                      max_iters=args.k, 
                                      _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    # 數據加載（與main.py相同）
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
    ])
    if args.todo == 'train':
        tr_dataset = tv.datasets.MNIST(args.data_root, 
                                      train=True, 
                                      transform=transform, 
                                      download=True)
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)

        te_dataset = tv.datasets.MNIST(args.data_root, 
                                      train=False, 
                                      transform=transform, 
                                      download=True)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
        save_model(model, os.path.join(model_folder, 'model2.pth'))  # 修改6：保存文件名區分

    elif args.todo == 'test':
        # 修改7：實現與main.py相同的測試流程
        te_dataset = tv.datasets.MNIST(args.data_root, 
                                      train=False, 
                                      transform=transform, 
                                      download=True)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False)
        
        if os.path.exists(os.path.join(model_folder, 'model2.pth')):
            model.load_state_dict(torch.load(os.path.join(model_folder, 'model2.pth')))
            test_acc, robust_acc = trainer.test(model, te_loader, True)
            logger.info('\n' + '='*30 + ' Final Test ' + '='*30)
            logger.info('Standard Accuracy: %.3f%%' % (test_acc * 100))
            logger.info('Robust Accuracy: %.3f%%' % (robust_acc * 100))
            logger.info('='*30 + ' Test Complete ' + '='*30 + '\n')
        else:
            logger.error('Error: Model file not found!')

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)

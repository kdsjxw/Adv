import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from src.time import time
from src.model import Model as Model1  # 修改1：明確導入原始模型為Model1
from src.model2 import Model as Model2  # 修改2：新增導入第二個模型
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model
from src.argument import parser, print_args

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def train(self, model1, model2, tr_loader, va_loader=None, adv_train=False):  # 修改3：接收兩個模型參數
        args = self.args
        logger = self.logger

        # 修改4：為兩個模型分別創建優化器
        opt1 = torch.optim.Adam(model1.parameters(), args.learning_rate)
        opt2 = torch.optim.Adam(model2.parameters(), args.learning_rate)

        _iter = 0
        begin_time = time()

        for epoch in range(1, args.max_epoch+1):
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                # 修改5：同時計算兩個模型的輸出
                if adv_train:
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    output1 = model1(adv_data, _eval=False)
                    output2 = model2(adv_data, _eval=False)
                else:
                    output1 = model1(data, _eval=False)
                    output2 = model2(data, _eval=False)

                # 修改6：分別計算損失
                loss1 = F.cross_entropy(output1, label)
                loss2 = F.cross_entropy(output2, label)

                # 修改7：分別反向傳播
                opt1.zero_grad()
                loss1.backward()
                opt1.step()

                opt2.zero_grad()
                loss2.backward()
                opt2.step()

                if _iter % args.n_eval_step == 0:
                    # 修改8：同時評估兩個模型
                    with torch.no_grad():
                        std_output1 = model1(data, _eval=True)
                        std_output2 = model2(data, _eval=True)
                    
                    pred1 = torch.max(std_output1, dim=1)[1]
                    pred2 = torch.max(std_output2, dim=1)[1]
                    
                    std_acc1 = evaluate(pred1.cpu().numpy(), label.cpu().numpy()) * 100
                    std_acc2 = evaluate(pred2.cpu().numpy(), label.cpu().numpy()) * 100

                    logger.info('epoch: %d, iter: %d, spent %.2f s' % (
                        epoch, _iter, time() - begin_time))
                    logger.info('model1 - std_acc: %.3f%%, model2 - std_acc: %.3f%%' % (
                        std_acc1, std_acc2))

                    if va_loader is not None:
                        va_acc1, _ = self.test(model1, va_loader, False)
                        va_acc2, _ = self.test(model2, va_loader, False)
                        logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                        logger.info('model1 - test_acc: %.3f%%, model2 - test_acc: %.3f%%' % (
                            va_acc1*100, va_acc2*100))
                        logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time()

                _iter += 1

    def test(self, model, loader, adv_test=False):
        # 保持原有測試邏輯不變
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

    # 修改9：同時初始化兩個模型
    model1 = Model1(i_c=1, n_c=10)
    model2 = Model2(i_c=1, n_c=10)  # 假設Model2有相同接口

    attack = FastGradientSignUntargeted(model1,  # 修改10：攻擊對象暫時用model1
                                      args.epsilon, 
                                      args.alpha, 
                                      min_val=0, 
                                      max_val=1, 
                                      max_iters=args.k, 
                                      _type=args.perturbation_type)

    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()

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

        # 修改11：傳入兩個模型進行訓練
        trainer.train(model1, model2, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        pass
    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)


import os
import torch
import torch.nn as nn

import torchvision as tv

from time import time
from torch.utils.data import DataLoader
from src.model.madry_model import WideResNet
from src.model.model2 import Model2
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, evaluate, save_model

from src.argument import parser, print_args

class EnsembleTrainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack
        self.models = {
            'madry': WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0),
            'model2': Model2(num_classes=10, drop_rate=0.1)
        }
        
    def _parallel_attack(self, data, label, mode='mean'):
        """同時生成對兩個模型有效的對抗樣本"""
        adv_data = {}
        for name, model in self.models.items():
            self.attack.model = model  # 切換攻擊目標模型
            adv_data[name] = self.attack.perturb(data, label, mode, True)
        return adv_data

    def train(self, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        # 雙模型優化器配置
        optimizers = {
            'madry': torch.optim.SGD(
                self.models['madry'].parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            ),
            'model2': torch.optim.AdamW(
                self.models['model2'].parameters(),
                lr=args.learning_rate * 0.5,
                weight_decay=args.weight_decay
            )
        }

        schedulers = {
            'madry': torch.optim.lr_scheduler.MultiStepLR(
                optimizers['madry'],
                milestones=[40000, 60000],
                gamma=0.1
            ),
            'model2': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers['model2'],
                T_max=args.max_epoch * len(tr_loader)
            )
        }

        _iter = 0
        begin_time = time()

        for epoch in range(1, args.max_epoch + 1):
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                # 同步訓練流程
                losses = {}
                for name, model in self.models.items():
                    model.train()
                    optimizers[name].zero_grad()

                    # 對抗訓練分支
                    if adv_train:
                        adv_data = self._parallel_attack(data, label)
                        output = model(adv_data[name], _eval=False)
                    else:
                        output = model(data, _eval=False)

                    loss = F.cross_entropy(output, label)
                    losses[name] = loss
                    loss.backward()
                    
                    # 梯度裁剪策略差異化
                    if name == 'madry':
                        nn.utils.clip_grad_value_(model.parameters(), 2.0)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    
                    optimizers[name].step()
                    schedulers[name].step()

                # 聯合評估（每n_eval_step迭代）
                if _iter % args.n_eval_step == 0:
                    eval_results = {}
                    with torch.no_grad():
                        for name, model in self.models.items():
                            model.eval()
                            # 標準準確率
                            std_output = model(data, _eval=True)
                            std_acc = evaluate(std_output.argmax(1).cpu(), label.cpu()) * 100
                            
                            # 對抗準確率
                            if adv_train:
                                adv_acc = evaluate(output.argmax(1).cpu(), label.cpu()) * 100
                            else:
                                adv_data = self.attack.perturb(data, label, 'mean', False)
                                adv_output = model(adv_data, _eval=True)
                                adv_acc = evaluate(adv_output.argmax(1).cpu(), label.cpu()) * 100
                            
                            eval_results[name] = (std_acc, adv_acc)
                            model.train()

                    # 日誌記錄
                    logger.info(f'[Epoch {epoch} Iter {_iter}]')
                    for name, (std_acc, adv_acc) in eval_results.items():
                        logger.info(
                            f'{name:6s} | lr: {schedulers[name].get_last_lr()[0]:.2e} | '
                            f'loss: {losses[name].item():.3f} | '
                            f'std_acc: {std_acc:.2f}% | adv_acc: {adv_acc:.2f}%'
                        )
                    logger.info(f'Time elapsed: {time()-begin_time:.1f}s\n')
                    begin_time = time()

                # 檢查點保存
                if _iter % args.n_checkpoint_step == 0:
                    for name, model in self.models.items():
                        save_model(
                            model,
                            os.path.join(args.model_folder, f'{name}_iter_{_iter}.pth')
                        )

                _iter += 1

            # 每epoch驗證
            if va_loader is not None:
                va_results = self.test(va_loader, adv_test=True)
                logger.info('\n' + '='*30 + f' Epoch {epoch} Validation ' + '='*30)
                for name, (std_acc, adv_acc) in va_results.items():
                    logger.info(f'{name:6s} | val_std: {std_acc:.2f}% | val_adv: {adv_acc:.2f}%')
                logger.info('='*28 + ' Validation End ' + '='*28 + '\n')

    def test(self, loader, adv_test=False):
        results = {}
        with torch.no_grad():
            for name, model in self.models.items():
                model.eval()
                total_std, total_adv, num = 0, 0, 0
                
                for data, label in loader:
                    data, label = tensor2cuda(data), tensor2cuda(label)
                    std_output = model(data, _eval=True)
                    total_std += evaluate(std_output.argmax(1).cpu(), label.cpu(), 'sum')
                    
                    if adv_test:
                        adv_data = self.attack.perturb(data, label, 'mean', False)
                        adv_output = model(adv_data, _eval=True)
                        total_adv += evaluate(adv_output.argmax(1).cpu(), label.cpu(), 'sum')
                    num += len(data)
                
                results[name] = (
                    total_std / num * 100,
                    (total_adv / num * 100) if adv_test else -1
                )
        return results

def main(args):
    # 路徑設置（添加ensemble標記）
    save_folder = f'{args.dataset}_{args.affix}_ensemble'
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)
    
    makedirs(log_folder)
    makedirs(model_folder)
    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')
    print_args(args, logger)

    # 初始化攻擊模組（默認使用madry_model作為攻擊目標）
    attack = FastGradientSignUntargeted(
        None,  # 實際攻擊時動態替換
        args.epsilon,
        args.alpha,
        min_val=0,
        max_val=1,
        max_iters=args.k,
        _type=args.perturbation_type
    )

    trainer = EnsembleTrainer(args, logger, attack)

    # 數據加載（與原有實現一致）
    transform_train = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
    ])
    transform_test = tv.transforms.ToTensor()

    if args.todo == 'train':
        tr_dataset = tv.datasets.CIFAR10(args.data_root, train=True, transform=transform_train, download=True)
        te_dataset = tv.datasets.CIFAR10(args.data_root, train=False, transform=transform_test, download=True)
        
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)
        
        trainer.train(tr_loader, te_loader, args.adv_train)

    elif args.todo == 'test':
        te_dataset = tv.datasets.CIFAR10(args.data_root, train=False, transform=transform_test, download=True)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)
        
        # 加載雙模型檢查點
        for name in ['madry', 'model2']:
            checkpoint = torch.load(os.path.join(args.load_checkpoint, f'{name}_final.pth'))
            trainer.models[name].load_state_dict(checkpoint)
        
        results = trainer.test(te_loader, adv_test=True)
        for name, (std_acc, adv_acc) in results.items():
            logger.info(f'{name:6s} | test_std: {std_acc:.2f}% | test_adv: {adv_acc:.2f}%')

    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)

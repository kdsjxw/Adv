import os
import torch
import torch.nn as nn

import torchvision as tv

from time import time
from torch.utils.data import DataLoader
from src.model.model2 import Model2  # 使用互補模型
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, evaluate, save_model

from src.argument import parser, print_args

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        # 優化器調整：適用於深度可分離卷積的較低學習率
        opt = torch.optim.AdamW(model.parameters(), 
                              lr=args.learning_rate * 0.5,  # 因模型輕量化，學習率減半
                              weight_decay=args.weight_decay)
        
        # 動態學習率調整（反向殘差需要更平滑的衰減）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.max_epoch * len(tr_loader), eta_min=1e-5
        )

        _iter = 0
        begin_time = time()

        for epoch in range(1, args.max_epoch + 1):
            model.train()
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                # 對抗訓練時採用混合精度攻擊
                if adv_train:
                    with torch.cuda.amp.autocast():  # 節省顯存
                        adv_data = self.attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)

                opt.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止深度可分離卷積的梯度爆炸）
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt.step()
                scheduler.step()

                # 評估邏輯（每n_eval_step迭代）
                if _iter % args.n_eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        if adv_train:
                            std_output = model(data, _eval=True)
                            std_acc = evaluate(std_output.argmax(1).cpu(), label.cpu()) * 100
                            adv_acc = evaluate(output.argmax(1).cpu(), label.cpu()) * 100
                        else:
                            adv_data = self.attack.perturb(data, label, 'mean', False)
                            adv_output = model(adv_data, _eval=True)
                            adv_acc = evaluate(adv_output.argmax(1).cpu(), label.cpu()) * 100
                            std_acc = evaluate(output.argmax(1).cpu(), label.cpu()) * 100

                    logger.info(f'epoch: {epoch}, iter: {_iter}, lr: {scheduler.get_last_lr()[0]:.2e}, '
                              f'time: {time()-begin_time:.1f}s, loss: {loss.item():.3f}')
                    logger.info(f'std_acc: {std_acc:.2f}%, adv_acc: {adv_acc:.2f}%')
                    begin_time = time()
                    model.train()

                # 保存檢查點和可視化
                if _iter % args.n_store_image_step == 0 and adv_train:
                    tv.utils.save_image(
                        torch.cat([data[:8].cpu(), adv_data[:8].cpu()], dim=0),
                        os.path.join(args.log_folder, f'adv_examples_{_iter}.png'),
                        nrow=8
                    )

                if _iter % args.n_checkpoint_step == 0:
                    save_model(model, os.path.join(args.model_folder, f'iter_{_iter}.pth'))

                _iter += 1

            # 每epoch結束驗證
            if va_loader is not None:
                va_acc, va_adv_acc = self.test(model, va_loader, adv_test=True)
                logger.info('\n' + '='*30 + f' Epoch {epoch} Validation ' + '='*30)
                logger.info(f'val_std_acc: {va_acc:.2f}%, val_adv_acc: {va_adv_acc:.2f}%')
                logger.info('='*28 + ' Validation End ' + '='*28 + '\n')

    def test(self, model, loader, adv_test=False):
        model.eval()
        total_std, total_adv, num = 0, 0, 0
        
        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                std_output = model(data, _eval=True)
                total_std += evaluate(std_output.argmax(1).cpu(), label.cpu(), 'sum')
                
                if adv_test:
                    adv_data = self.attack.perturb(data, label, 'mean', False)
                    adv_output = model(adv_data, _eval=True)
                    total_adv += evaluate(adv_output.argmax(1).cpu(), label.cpu(), 'sum')
                num += len(data)

        return total_std / num * 100, (total_adv / num * 100) if adv_test else -1

def main(args):
    # 路徑設置（與main.py一致）
    save_folder = f'{args.dataset}_{args.affix}_model2'
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)
    
    makedirs(log_folder)
    makedirs(model_folder)
    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')
    print_args(args, logger)

    # 使用Model2並調整攻擊參數（因模型結構差異）
    model = Model2(num_classes=10, drop_rate=0.1)
    attack = FastGradientSignUntargeted(
        model,
        epsilon=args.epsilon * 0.8,  # 更小的攻擊步長（因模型輕量化）
        alpha=args.alpha * 0.5,
        min_val=0,
        max_val=1,
        max_iters=args.k,
        _type=args.perturbation_type
    )

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    # 數據加載（保持與main.py相同的預處理）
    transform_train = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
    ])
    transform_test = tv.transforms.ToTensor()

    if args.todo == 'train':
        tr_dataset = tv.datasets.CIFAR10(args.data_root, train=True, transform=transform_train, download=True)
        te_dataset = tv.datasets.CIFAR10(args.data_root, train=False, transform=transform_test, download=True)
        
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)
        
        trainer.train(model, tr_loader, te_loader, args.adv_train)

    elif args.todo == 'test':
        te_dataset = tv.datasets.CIFAR10(args.data_root, train=False, transform=transform_test, download=True)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)
        
        model.load_state_dict(torch.load(args.load_checkpoint))
        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True)
        
        logger.info(f'Final Test - std_acc: {std_acc:.2f}%, adv_acc: {adv_acc:.2f}%')
    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)

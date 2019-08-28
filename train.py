import torch
import torch.utils.data
from sample.ctdet import CTDetDataset
from datasets.cocodataset import COCO
from utils.opts import opts
from Pytorch_model.model import create_model
from Pytorch_model.model import load_model
from trains.ctdet import CtdetTrainer
import os
from collections import deque
from Pytorch_model.model import save_model
from utils.opts import opts

def get_dataset():
  class Dataset(COCO, CTDetDataset):
    pass
  return Dataset

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    Dataset = get_dataset()
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt.device = 'cuda'

    print('Creating model...')
    model = create_model(18, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    Trainer = CtdetTrainer
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    best = 1e10
    losses = deque(maxlen=1000)

    if not os.path.exists(os.path.join(opt.root_dir, 'exp')):
        os.mkdir(os.path.join(opt.root_dir, 'exp'))
    if not os.path.exists(opt.exp_dir):
        os.mkdir(opt.exp_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)



    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        log_dict_train, _ = trainer.train(epoch, train_loader, losses)
        avg_loss = sum(losses) / len(losses)
        print('epoch: {}, loss: {} |'.format(epoch, avg_loss))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader, losses)
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == '__main__':
    opt = opts().init()
    main(opt)

import os
import time
import torch.optim as optim
from tqdm import tqdm
from torchutil.EarlyStopping import *
from torchutil.meter import *
import torchutil
import torchio as tio
from .loss import *
from .UNet3D import *
from .MixUNet3D import *
from .VNet import *
# from .UNet import *
# from .NestedUNet import *
# from .UNetP3d import UNetP3D
from .utils import split, combine, PolyLrDecay
from datasets.DataLoaderX import DataLoaderX
from torch.cuda.amp import autocast as autocast, GradScaler
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def DiceV(A, B):
    intersection = torch.logical_and(A, B)
    eps = 1e-5
    return ((2 * intersection.sum() + eps) / (A.sum() + B.sum() + eps)).item()


class DPModel(nn.Module):
    def __init__(self, config):
        super(DPModel, self).__init__()
        self.config = config
        network_params = config['network']
        self.model_name = network_params['model_name']
        self.device = torch.device(network_params['device'])
        self.net = globals()[network_params['model_name']](config)
        
        if network_params['load_pretrained']:
            checkpoint = torch.load(network_params['pretrained_path'])
            self.net.load_state_dict(checkpoint)
        if network_params['num_gpus'] > 1:
            self.net = nn.DataParallel(self.net).to(self.device)
        else:
            self.net = self.net.to(self.device)
        self.logger = None
        self.ckpt_path = None
        self.ddice = DDice()

    def train(self, train_loader, eval_loader):
        optim_params = self.config['optim']
        if optim_params['optim_method'].lower() == 'sgd':
            sgd_params = optim_params['sgd']
            optimizer = optim.SGD(self.net.parameters(),
                                  lr=sgd_params['base_lr'],
                                  momentum=sgd_params['momentum'],
                                  weight_decay=sgd_params['weight_decay'],
                                  nesterov=sgd_params['nesterov'])
        elif optim_params['optim_method'].lower() == 'adam':
            adam_params = optim_params['adam']
            optimizer = optim.Adam(self.net.parameters(),
                                   lr=adam_params['base_lr'],
                                   betas=adam_params['betas'],
                                   weight_decay=adam_params['weight_decay'],
                                   amsgrad=adam_params['amsgrad'])
        else:
            raise Exception('Not support optim method: {}.'.format(optim_params['optim_method']))
        # choosing whether to use lr_decay and related parameters
        if optim_params['use_lr_decay']:
            from torch.optim import lr_scheduler
            if optim_params['lr_decay_method'] == 'cosine':
                lr_decay = lr_scheduler.CosineAnnealingLR(
                    optimizer, eta_min=0, T_max=optim_params['num_epochs'])
            elif optim_params['lr_decay_method'] == 'poly':
                lr_decay = PolyLrDecay(optimizer, 
                                       optim_params['num_epochs'], 
                                       optim_params["poly"]["exponent"])
        logging_params = self.config['logging']
        self.ckpt_path = logging_params['ckpt_path']
        if logging_params['use_logging']:

            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self.logger = torchutil.log.get_logger(os.path.join(self.ckpt_path, '{}.log'.format(self.model_name)))
            nums = get_parameter_number(self.net)
            self.logger.info(">>>num of parameters in the net is:")
            self.logger.info(f"total num = {nums[0]}, trainable num = {nums[1]}")
            self.logger.info(">>>The net is:")
            self.logger.info(self.net)
            self.logger.info(">>>The config is:")
            self.logger.info(torchutil.format_config(self.config))

        criterions = [criterion.to(self.device) for criterion in get_criterion(self.config['criterion'])]
        best_evals = {'epoch': 0, 'metrics': {'loss': 1000.}}
        metric_name = 'loss'

        estop = torchutil.EarlyStopping(mode='min', patience=self.config["optim"]["early_stop"]["patience"])
        scaler = GradScaler()
        for epoch_id in range(optim_params['num_epochs']):
            self.train_epoch(epoch_id, train_loader, criterions, optimizer, scaler)
            if optim_params['use_lr_decay']:
                lr_decay.step()
                print("now lr is:", lr_decay.get_last_lr())
            test_metric = self.test_epoch(epoch_id, eval_loader, criterions)

            # saving the best model
            # if mse_eval_loss <= best_evals[-1]:
            if best_evals['metrics'][metric_name] >= test_metric[metric_name]:
                best_evals['metrics'] = test_metric
                best_evals['epoch'] = epoch_id
                self.save(epoch_id)
            self.logger.info(
                '[Info]epoch=[{}] minimal loss=[{:.5f}] '.format(best_evals['epoch'],
                                                                 best_evals['metrics'][metric_name]))
            self.logger.info('save to {}'.format(self.ckpt_path))
            if estop.step(test_metric[metric_name]) and self.config["optim"]["early_stop"]["use"]:
                self.logger.rm_log()
                break  # early stop criterion is met, we can stop now
        pass

    def train_epoch(self, epoch_id, data_loader, criterions, optimizer, scaler=None):
        loss_meter = torchutil.AverageMeter()
        self.net.train()

        with tqdm(total=len(data_loader)) as pbar:
            for batch_id, sample in enumerate(data_loader):
                #print(sample['image'][tio.DATA].shape, sample['label'][tio.DATA].shape)
                image = sample['image'][tio.DATA].to(self.device)
                label = torch.squeeze(sample['label'][tio.DATA], dim=1).float().to(self.device)
                optimizer.zero_grad()
                with autocast():
                    logits = self.net(image)
                    coefs = self.config['criterion']['criterion_coefs']

                    if isinstance(logits, list) and self.config["network"]["deep_supervision"]:
                        loss = 0
                        for i, logit in enumerate(logits):
                            wl, wt = list(logit.shape), list(label.shape)
                            #print(wl, wt, torch.unsqueeze(label, dim=1).size())
                            if np.sum(np.abs(np.array(wl) - np.array(wt))) != 0:
                                #wl.insert(1,1)
                                with torch.no_grad():
                                    target  = torch.nn.functional.interpolate(torch.unsqueeze(label, dim=1), 
                                                                                size=tuple(wl[1:]), mode='nearest')
                                    target = torch.squeeze(target, dim=1)
                            else:
                                target = label
                            loss += (1./np.power(2, i))*sum([criterion(logit, target) * coef for 
                                         criterion, coef in zip(criterions, coefs)])
                    else:
                        loss = sum([criterion(logits, label) * coef for criterion, coef in zip(criterions, coefs)])
                #loss.backward()
                #optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_meter.update(loss.item(), image.size(0))
                pbar.update(1)
                pbar.set_description("[Train] Epoch=[{}], Loss=[{:.5f}]".format(epoch_id, loss_meter.avg))

        logging_params = self.config['logging']
        if logging_params['use_logging']:
            self.logger.info("[Train] Epoch=[{}], Loss=[{:.5f}]".format(epoch_id, loss_meter.avg))
        return {'loss': loss_meter.avg}

    def test_epoch(self, epoch_id, data_loader, criterions, rtn=False):
        dices = []
        loss_meter = torchutil.AverageMeter()
        self.net.eval()
        with torch.no_grad():
            with tqdm(total=len(data_loader)) as pbar:
                for batch_id, sample in enumerate(data_loader):
                    grid_sampler = tio.inference.GridSampler(sample,
                                                             data_loader.patch_size,
                                                             data_loader.patch_overlap,
                                                             #padding_mode=0,
                                                            )
                    patch_loader = DataLoaderX(grid_sampler, 
                                               batch_size=data_loader.batch_size, 
                                               num_workers=2)
                    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')
                    for patches_batch in patch_loader:
                        with autocast():
                            img_corps = patches_batch['image'][tio.DATA].to(self.device)
                            locations = patches_batch[tio.LOCATION]
                            pred = self.net(img_corps)
                            if isinstance(pred, list):
                                pred = torch.sigmoid(pred[0])
                            else:
                                pred = torch.sigmoid(pred)
                            aggregator.add_batch(torch.unsqueeze(pred, dim=1), locations)
                    final_pred = aggregator.get_output_tensor()
                    label = sample['label'][tio.DATA]#.to(self.device)
                    d = DiceV(final_pred > 0.5, label)
                    dices.append(d)
                    loss_meter.update(1-d, 1)
                    pbar.update(1)
                    pbar.set_description("[Eval] Epoch=[{}], Loss=[{:.5f}]".format(epoch_id, loss_meter.avg))
                    del final_pred

            logging_params = self.config['logging']
            if logging_params['use_logging']:
                self.logger.info("[Eval] Epoch=[{}], Loss=[{:.5f}]".format(epoch_id, loss_meter.avg))
        if rtn:
            return {'loss': loss_meter.avg}, dices
        else:
            return {'loss': loss_meter.avg}
    
    def eval_epoch(self, epoch_id, data_loader, criterions, rtn=False):
        dices = []
        loss_meter = torchutil.AverageMeter()
        self.net.eval()
        with torch.no_grad():
            with tqdm(total=len(data_loader)) as pbar:
                for batch_id, sample in enumerate(data_loader):
                    grid_sampler = tio.inference.GridSampler(sample,
                                                             data_loader.patch_size,
                                                             data_loader.patch_overlap,
                                                             #padding_mode=0,
                                                            )
                    patch_loader = DataLoaderX(grid_sampler, 
                                               batch_size=data_loader.batch_size, 
                                               num_workers=2)
                    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')
                    for patches_batch in patch_loader:
                        with autocast():
                            img_corps = patches_batch['image'][tio.DATA].to(self.device)
                            locations = patches_batch[tio.LOCATION]
                            pred = self.net(img_corps)
                            if isinstance(pred, list):
                                pred = torch.sigmoid(pred[0])
                            else:
                                pred = torch.sigmoid(pred)
                            aggregator.add_batch(torch.unsqueeze(pred, dim=1), locations)
                    final_pred = aggregator.get_output_tensor()
                    label = sample['label'][tio.DATA]#.to(self.device)
                    d = DiceV(final_pred > 0.5, label)
                    dices.append(d)
                    loss_meter.update(1-d, 1)
                    pbar.update(1)
                    pbar.set_description("[Eval] Epoch=[{}], Loss=[{:.5f}]".format(epoch_id, loss_meter.avg))
                    del final_pred

            logging_params = self.config['logging']
            if logging_params['use_logging']:
                self.logger.info("[Eval] Epoch=[{}], Loss=[{:.5f}]".format(epoch_id, loss_meter.avg))
        if rtn:
            return {'loss': loss_meter.avg}, dices
        else:
            return {'loss': loss_meter.avg}

    def save(self, epoch):
        if self.config['network']['num_gpus'] > 1:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, os.path.join(self.ckpt_path, '{}.pth'.format("best")))


def get_criterion(criterion_params):
    # choosing criterion
    criterions = []
    for i, criterion_name in enumerate(criterion_params['criterions']):
        if criterion_name.upper() == 'CE_LOSS':
            loss_params = criterion_params[criterion_name]
            weight = torch.Tensor(loss_params['weight'])
            criterion = CELoss(weight=weight)
        elif criterion_name.upper() == 'DICE_LOSS':
            criterion = DiceLoss()
        elif criterion_name.upper() == 'BCELOSS':
            criterion = BCELoss()
        elif criterion_name.upper() == 'DICE':
            criterion = Dice()
        else:
            raise Exception('Not support criterion method: {}.'.format(criterion_name))
        criterions.append(criterion)
    return criterions

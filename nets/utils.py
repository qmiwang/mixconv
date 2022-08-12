import numpy as np


def split(image, crop_size=[64, 128, 128], skip=[64, 128, 128]):
    start_idxs = []
    imgs = []
    z, x, y = image.shape[2:]
    zzs = [zz for zz in range(0, z - crop_size[0] + 1, skip[0])]
    if zzs[-1] + crop_size[0] < z:
        zzs.append(z - crop_size[0])
    for zz in zzs:
        for xx in range(0, x - crop_size[1] + 1, skip[1]):
            for yy in range(0, y - crop_size[2] + 1, skip[2]):
                start_idxs.append([zz, xx, yy])
                imgs.append(image[:, :, zz:zz + crop_size[0], xx:xx + crop_size[1], yy:yy + crop_size[2]])
    return imgs, start_idxs


def combine(preds, start_idxs, image_size):
    final_pred = np.zeros(image_size)
    masks = np.zeros(image_size)
    crop_size = [64, 128, 128]
    for i, (zz, xx, yy) in enumerate(start_idxs):
        final_pred[zz:zz + crop_size[0], xx:xx + crop_size[1], yy:yy + crop_size[2]] += np.squeeze(preds[i])
        masks[zz:zz + crop_size[0], xx:xx + crop_size[1], yy:yy + crop_size[2]] += 1
    final_pred = final_pred / masks
    return final_pred

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

class PolyLrDecay:
    def __init__(self, optimizer, max_epochs, exponent=0.9):
        self.optimizer = optimizer
        self.initial_lrs = list([])
        for param_group in self.optimizer.param_groups:
            self.initial_lrs.append(param_group['lr'])
        self.max_epochs = max_epochs
        self.exponent = exponent
        self.current_step = 0
    def step(self):
        self.current_step += 1
        for param_group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            param_group['lr'] = poly_lr(self.current_step, self.max_epochs, initial_lr, self.exponent)
    def reset(self):
        self.current_step = 0
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

import os
import torch
import shutil
from collections import OrderedDict
import logging
import numpy as np


# For model related utils
def save_ckpt(model, optimizer, loss, epoch, save_path, name_pre, name_post='best'):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
            'epoch': epoch,
            'model_state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory ", save_path, " is created.")

    filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    print('model has been saved as {}'.format(filename))


def load_pretrained_models(model, pretrained_model, ismax=True, printer=logging.info):  # ismax means max best
    if ismax:
        best_value = -np.inf
    else:
        best_value = np.inf
    epoch = 0

    if pretrained_model:
        if os.path.isfile(pretrained_model):
            printer("===> Loading checkpoint '{}'".format(pretrained_model))
            checkpoint = torch.load(pretrained_model)

            best_value_key = [key for key in checkpoint.keys() if 'best' in key.lower()]
            best_value_key = None if len(best_value_key) == 0 else best_value_key[0]

            if best_value_key is not None:
                best_value = checkpoint[best_value_key]
                if best_value == -np.inf or best_value == np.inf:
                    show_best_value = False
                else:
                    show_best_value = True
            else:
                show_best_value = False

            model_dict = model.state_dict()
            ckpt_model_state_dict = checkpoint['state_dict']

            # rename ckpt (avoid name is not same because of multi-gpus)
            is_model_multi_gpus = True if list(model_dict)[0].split('.')[0] == 'module' else False
            is_ckpt_multi_gpus = True if list(ckpt_model_state_dict)[0].split('.')[0] == 'module' else False

            if not (is_model_multi_gpus == is_ckpt_multi_gpus):
                temp_dict = OrderedDict()
                for k, v in ckpt_model_state_dict.items():
                    if is_ckpt_multi_gpus:
                        name = k[7:]  # remove 'module.'
                    else:
                        name = 'module.'+k  # add 'module'
                    temp_dict[name] = v
                # load params
                ckpt_model_state_dict = temp_dict

            model_dict.update(ckpt_model_state_dict)
            model.load_state_dict(ckpt_model_state_dict)

            if 'epoch' in checkpoint.keys():
                epoch = checkpoint['epoch']

            if show_best_value:
                logging.info(f"The pretrained model is at checkpoint {epoch}. \t "
                             f"{best_value_key}: {best_value}")
            else:
                logging.info("The pretrained_model is at checkpoint {}.".format(epoch))

        else:
            raise ImportError("===> No checkpoint found at '{}'".format(pretrained_model))
    else:
        printer('===> No pre-trained model is provided. ')
    return model, best_value, epoch


def load_pretrained_optimizer(pretrained_model, optimizer, scheduler, device='cuda'):
    if pretrained_model:
        if os.path.isfile(pretrained_model):
            checkpoint = torch.load(pretrained_model)

            optimizer_key = [key for key in checkpoint.keys() if 'opt' in key.lower()]
            optimizer_key = None if len(optimizer_key) == 0 else optimizer_key[0]

            scheduler_key = [key for key in checkpoint.keys() if 'scheduler' in key.lower()]
            scheduler_key = None if len(scheduler_key) == 0 else scheduler_key[0]

            if optimizer_key is not None:
                optimizer.load_state_dict(checkpoint[optimizer_key])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            if scheduler_key is not None:
                scheduler.load_state_dict(checkpoint[scheduler_key])

    return optimizer, scheduler


def save_checkpoint(state, is_best, save_path, postname):
    filename = '{}/{}_{}.pth'.format(save_path, postname, int(state['epoch']))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_best.pth'.format(save_path, postname))


def change_ckpt_dict(model, optimizer, scheduler, opt):

    for _ in range(opt.epoch):
        scheduler.step()
    is_best = (opt.test_value < opt.best_value)
    opt.best_value = min(opt.test_value, opt.best_value)

    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    # optim_cpu = {k: v.cpu() for k, v in optimizer.state_dict().items()}
    save_checkpoint({
        'epoch': opt.epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_value': opt.best_value,
    }, is_best, opt.save_path, opt.post)


def get_model_params(model, bits=32):
    """Gets the model parameter size. One can also refer to torchsummay.
    Args:
        model: the input network
        bits: the default bits (torch.float32), 32 bits

    Returns:
        the size of model (Mb)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * (bits/8) / (1024 * 1024)


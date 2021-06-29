import torch
import os


def save_checkpoint(netD, netG, dir_path, epoch, name, FLAGS):
    path = dir_path + "_" + name + "_" + FLAGS.dataset

    if not os.path.exists(dir_path + "_" + name):
        os.makedirs(dir_path + "_" + name)

    torch.save(netD.state_dict(), '{0}/{2}_disc_{1}.pth'.format(path, epoch, name))
    torch.save(netG.state_dict(), '{0}/{2}_gen_{1}.pth'.format(path, epoch, name))



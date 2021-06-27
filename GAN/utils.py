import torch
import os


def save_checkpoint(netD, netG, dir_path, epoch):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(dir_path, epoch))
    torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(dir_path, epoch))



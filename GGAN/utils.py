import torch
import os


def save_checkpoint(netD, netG, dir_path, epoch, FLAGS):
    path = dir_path + FLAGS.experiment_name

    if not os.path.exists(dir_path + FLAGS.experiment_name):
        os.makedirs(dir_path + FLAGS.experiment_name)

    torch.save(netD.state_dict(), '{0}/{2}_disc_{1}.pth'.format(path, epoch, FLAGS.experiment_name))
    torch.save(netG.state_dict(), '{0}/{2}_gen_{1}.pth'.format(path, epoch, FLAGS.experiment_name))



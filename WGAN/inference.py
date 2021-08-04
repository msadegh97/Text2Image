import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import  make_grid
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader
from easydict import EasyDict
from texttoimageloader import Text2ImageDataset
from WGAN import Generator
import os

FLAGS = flags.FLAGS

flags.DEFINE_integer('ngpu', 1, 'number of GPUs')
flags.DEFINE_string('dataset', 'flowers', 'name of dataset (birds or flowers)')
flags.DEFINE_integer('z_dim', 100, 'Input noise dimension')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('lr', 0.0002, 'learning rate')
flags.DEFINE_float('beta', 0.5, 'beta')
flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
flags.DEFINE_integer('num_workers', 2, 'number of workers')
flags.DEFINE_bool('cls', True, 'add wrong image loss')
flags.DEFINE_string("checkpoints_path", './models/', 'checkpoints_path')
flags.DEFINE_integer("critic_repeats", 5, 'critic opt / generator opt')
flags.DEFINE_float("labmda1",10.0, "Gradient Penalty Coef")
flags.DEFINE_integer("embed_dim", 1024, "text embedding dim")
flags.DEFINE_integer("proj_embed_dim", 256, "projected text embedding dim")

flags.DEFINE_integer("cp_interval", 10, 'checkpoint intervals (epochs)')
flags.DEFINE_integer("log_interval", 10, 'log intervals (steps)')

flags.DEFINE_bool("wandb", False, "Using wandb for logging")
flags.DEFINE_string('wandb_key', '', 'wandb key for logging')


flags.DEFINE_string("pre_trained_critic", '', 'pretrained critic path')
flags.DEFINE_string("pre_trained_generator", '', 'pretrained generator path')


def predict(model, data_loader, dim_z):
    num_sam = 0
    for sample in data_loader:
        right_images = sample['right_images']
        right_embed = sample['right_embed']
        txt = sample['txt']


        right_images = torch.Variable(right_images.float()).cuda()
        right_embed = torch.Variable(right_embed.float()).cuda()

        # Train the generator
        noise = torch.Variable(torch.randn(6, dim_z)).cuda()
        noise = noise.view(noise.size(0), dim_z, 1, 1)
        fake_images = model(noise, right_embed)
        imges = torch.cat(right_images, fake_images)
        new_img = make_grid(imges, nrow=1, padding= 1)

        plt.title(txt)
        plt.imshow(new_img)
        plt.show()

        if not os.path.exists('./img'):
            os.makedirs("./img")

        plt.savefig(new_img, './img/'+str(num_sam)+'.png')

        num_sam +=1
        if num_sam == 10:
            break


def main():
    flags_dict = EasyDict()

    for key in dir(FLAGS):
        flags_dict[key] = getattr(FLAGS, key)


    dataset_add = '../dataset/'
    ## prepare Data
    if flags_dict.dataset == 'birds':
        dataset = Text2ImageDataset(dataset_add+'birds.hdf5', split=1)  ##TODO split
    elif flags_dict.dataset == 'flowers':
        dataset = Text2ImageDataset(dataset_add+ 'flowers.hdf5', split=1) ##TODO split
    else:
        raise ('Dataset not found')

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=flags_dict.num_workers)

    ##init modedls

    generator = Generator(z_dim=flags_dict.z_dim, proj_ebmed_dim=flags_dict.proj_embed_dim, embed_dim=flags_dict.embed_dim)
    generator.load_state_dict(torch.load(flags_dict.pre_trained_generator))
    generator.eval()
    predict(generator.cuda(),data_loader,100)




if __name__ == '__main__':
    main()









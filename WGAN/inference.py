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
import torchvision.transforms as transforms
from torch.autograd import Variable

from datasets import TextDataset
from datasets import prepare_data
import numpy as np
from DAMSM import RNN_ENCODER
from PIL import Image


FLAGS = flags.FLAGS

flags.DEFINE_integer('ngpu', 1, 'number of GPUs')
flags.DEFINE_string('dataset', 'flowers', 'name of dataset (birds or flowers)')
flags.DEFINE_integer('z_dim', 100, 'Input noise dimension')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_float('beta', 0.5, 'beta')
flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
flags.DEFINE_integer('num_workers', 2, 'number of workers')
flags.DEFINE_bool('cls', True, 'add wrong image loss')
flags.DEFINE_string("checkpoints_path", './models/', 'checkpoints_path')
flags.DEFINE_integer("critic_repeats", 5, 'critic opt / generator opt')
flags.DEFINE_float("labmda1",10.0, "Gradient Penalty Coef")
flags.DEFINE_integer("embed_dim", 256, "text embedding dim")
flags.DEFINE_integer("proj_embed_dim", 256, "projected text embedding dim")
flags.DEFINE_string("img_save_dir", './img/', 'save images folder')

flags.DEFINE_integer("cp_interval", 10, 'checkpoint intervals (epochs)')
flags.DEFINE_integer("log_interval", 10, 'log intervals (steps)')

flags.DEFINE_bool("wandb", False, "Using wandb for logging")
flags.DEFINE_string('wandb_key', '', 'wandb key for logging')


flags.DEFINE_string("pre_trained_critic", '', 'pretrained critic path')
flags.DEFINE_string("gen_dir", '', 'pretrained generator path')

flags.DEFINE_string("experiment_name", 'exp', 'the experiment name')

def sample_img(FLAGS):
    split_dir = 'valid'
    imsize = 64
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset_add = '../dataset/'
    ## prepare Data
    if FLAGS.dataset == 'birds':
        dataset = TextDataset(dataset_add + "birds",
                              'test',
                              base_size=64,
                              transform=image_transform)
    else:
        raise ('Dataset not found')

    data_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers,
                             drop_last=True)

    # init emb model
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=256)
    state_dict = torch.load("../emb_model/bird/text_encoder200.pth", map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    netG = torch.nn.DataParallel(Generator(z_dim=FLAGS.z_dim, proj_ebmed_dim=FLAGS.proj_embed_dim, embed_dim=FLAGS.embed_dim).cuda(), range(FLAGS.ngpu))
    netG.load_state_dict(torch.load(FLAGS.gen_dir))
    netG.eval()
    batch_size = FLAGS.batch_size

    if not os.path.exists(FLAGS.img_save_dir + FLAGS.experiment_name):
        os.makedirs(FLAGS.img_save_dir + FLAGS.experiment_name)
    cnt = 0
    for i in range(1):
        for step, data in enumerate(data_loader, 0):
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)

            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
               # noise = torch.randn(batch_size, 100).cuda()
                
                noise = Variable(torch.randn(batch_size, 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_imgs = netG(noise, sent_emb)
            for j in range(batch_size):
                s_tmp =  FLAGS.img_save_dir + FLAGS.experiment_name + '/'+ str(j)+ "_" + str(step) #keys[j]
                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                #MEAN = torch.tensor([0.5, 0.5, 0.5])
                #STD = torch.tensor([0.5, 0.5, 0.5])
                # im = im * STD + MEAN
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                im.save(s_tmp+'.png', 'png')


#
#
# def predict(model, data_loader, dim_z):
#     num_sam = 0
#     for sample in data_loader:
#         right_images = sample['right_images']
#         right_embed = sample['right_embed']
#         txt = sample['txt']
#
#
#         right_images = torch.Variable(right_images.float()).cuda()
#         right_embed = torch.Variable(right_embed.float()).cuda()
#
#         # Train the generator
#         noise = torch.Variable(torch.randn(6, dim_z)).cuda()
#         noise = noise.view(noise.size(0), dim_z, 1, 1)
#         fake_images = model(noise, right_embed)
#         imges = torch.cat(right_images, fake_images)
#         new_img = make_grid(imges, nrow=1, padding= 1)
#
#         plt.title(txt)
#         plt.imshow(new_img)
#         plt.show()
#
#         if not os.path.exists('./img'):
#             os.makedirs("./img")
#
#         plt.savefig(new_img, './img/'+str(num_sam)+'.png')
#
#         num_sam +=1
#         if num_sam == 10:
#             break


def main():
    flags_dict = EasyDict()

    for key in dir(FLAGS):
        flags_dict[key] = getattr(FLAGS, key)


    # dataset_add = '../dataset/'
    # ## prepare Data
    # if flags_dict.dataset == 'birds':
    #     dataset = Text2ImageDataset(dataset_add+'birds.hdf5', split=1)  ##TODO split
    # elif flags_dict.dataset == 'flowers':
    #     dataset = Text2ImageDataset(dataset_add+ 'flowers.hdf5', split=1) ##TODO split
    # else:
    #     raise ('Dataset not found')
    #
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=flags_dict.num_workers)
    #
    # ##init modedls

    # generator = Generator(z_dim=flags_dict.z_dim, proj_ebmed_dim=flags_dict.proj_embed_dim, embed_dim=flags_dict.embed_dim)
    # generator.load_state_dict(torch.load(flags_dict.pre_trained_generator))
    # generator.eval()
    # predict(generator.cuda(),data_loader,100)

    sample_img(flags_dict)




if __name__ == '__main__':
    main()









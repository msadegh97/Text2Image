import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from easydict import EasyDict
from texttoimageloader import Text2ImageDataset
from DCGAN import Generator, Discriminator
import os
import wandb
from torchvision.utils import make_grid
import utils
from tensorflow.python.platform import flags
import torchvision.transforms as transforms
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from DAMSM import RNN_ENCODER


FLAGS = flags.FLAGS


flags.DEFINE_integer('ngpu', 1, 'number of GPUs')
flags.DEFINE_string('dataset', 'birds', 'name of dataset (birds or flowers)')
flags.DEFINE_integer('z_dim', 100, 'Input noise dimension')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_float('beta', 0.5, 'beta')
flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
flags.DEFINE_integer('num_workers', 2, 'number of workers')
flags.DEFINE_bool('cls', True, 'add wrong image loss')
flags.DEFINE_string("checkpoints_path", './models/', 'checkpoints_path')

flags.DEFINE_integer("embed_dim", 256, "text embedding dim")
flags.DEFINE_integer("proj_embed_dim", 256, "projected text embedding dim")

flags.DEFINE_integer("cp_interval", 10, 'checkpoint intervals (epochs)')
flags.DEFINE_integer("log_interval", 200, 'log intervals (steps)')

flags.DEFINE_bool("wandb", False, "Using wandb for logging")
flags.DEFINE_string('wandb_key', '', 'wandb key for logging')


flags.DEFINE_string("pre_trained_critic", '', 'pretrained critic path')
flags.DEFINE_string("pre_trained_generator", '', 'pretrained generator path')
flags.DEFINE_string("experiment_name", 'exp', 'the experiment name')
flags.DEFINE_bool("inter", True, "embedding interpolation")




FLAGS = flags.FLAGS


def train(FLAGS):
    if FLAGS.wandb:
        run = wandb.init(project="Text2Image_CON_DCGAN", config=FLAGS, name= FLAGS.experiment_name)

    imsize = 64
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset_add = '../dataset/'
    ## prepare Data
    if FLAGS.dataset == 'birds':
        dataset = TextDataset(dataset_add + "birds",
                              'train',
                              base_size=64,
                              transform=image_transform)
    else:
        raise ('Dataset not found')

    data_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers,  drop_last=True)


    #init emb model
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden= 256)
    state_dict = torch.load("../emb_model/bird/text_encoder200.pth", map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    ##init modedls
    generator = torch.nn.DataParallel(Generator(z_dim=FLAGS.z_dim, proj_ebmed_dim=FLAGS.proj_embed_dim, embed_dim=FLAGS.embed_dim).cuda(), range(FLAGS.ngpu))
    discriminator = torch.nn.DataParallel(Discriminator(proj_embed_dim=FLAGS.proj_embed_dim, embed_dim=FLAGS.embed_dim ).cuda(), range(FLAGS.ngpu))


    #set optimizers
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr= 4*FLAGS.lr, betas=(FLAGS.beta, 0.999))
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta, 0.999))


    criterion = nn.BCELoss()

    iteration = 0

    for epoch in range(FLAGS.num_epochs):
        for data in data_loader:
            iteration += 1
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            hidden = text_encoder.init_hidden(FLAGS.batch_size)

            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            real_image = imags[0].to('cuda')

            real_labels = torch.ones(real_image.size(0))
            fake_labels = torch.zeros(real_image.size(0))


            real_labels = Variable(real_labels).cuda()
            fake_labels = Variable(fake_labels).cuda()

            # Train the discriminator
            discriminator.zero_grad()
            outputs = discriminator(real_image, sent_emb)
            real_loss = criterion(outputs, real_labels)
            real_score = outputs

            if FLAGS.cls:
                outputs = discriminator(real_image, sent_emb.flip(0))
                wrong_loss = criterion(outputs, fake_labels)
                wrong_score = outputs

            noise = Variable(torch.randn(real_image.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = generator(noise, sent_emb)
            outputs = discriminator(fake_images.detach(), sent_emb)
            fake_loss = criterion(outputs, fake_labels)
            fake_score = outputs



            if FLAGS.cls:
                d_loss = real_loss + (fake_loss + wrong_loss) * 0.5
            else:
                d_loss = real_loss + fake_loss

            d_loss.backward()
            D_optimizer.step()

            # Train the generator
            generator.zero_grad()

            outputs = discriminator(fake_images, sent_emb)
            g_loss = criterion(outputs, real_labels)

            if FLAGS.inter:
                noise = Variable(torch.randn(real_image.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                int_embed = (sent_emb + sent_emb.flip(0))/2
                fake_images_int = generator(noise, int_embed)
                outputs_int = discriminator(fake_images_int, int_embed)
                g_loss = (criterion(outputs_int, real_labels) + g_loss)/2

            D_optimizer.zero_grad()
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            if FLAGS.wandb:
                if iteration % FLAGS.log_interval == 0:

                    real_image_grid = make_grid(real_image, nrow=8, pad_value=1)
                    fake_image_grid = make_grid(fake_images, nrow=8, pad_value=1)
                    _real_img = wandb.Image(real_image_grid, caption="real_images")
                    _fake_img = wandb.Image(fake_image_grid, caption="fake_images")

                    run.log({"Generator Loss": g_loss.item(),
                             "Discriminator Loss": d_loss.item(),
                             "Real Score": real_score.mean(),
                             "Fake Score": fake_score.mean(),
                             "real_img": _real_img,
                             "fake_img": _fake_img})


        print(epoch)
        if FLAGS.wandb:
            run.log({"Generator Loss_e": g_loss.item(),
                     "Discriminator Loss_e": d_loss.item(),
                     "Real Score_e": real_score.mean(),
                     "Fake Score_e": fake_score.mean(),
                     "epoch": epoch})

        if (epoch) % FLAGS.cp_interval == 0:
            utils.save_checkpoint(discriminator, generator, FLAGS.checkpoints_path, epoch, FLAGS)

def main():
    flags_dict = EasyDict()


    for key in dir(FLAGS):
        flags_dict[key] = getattr(FLAGS, key)
    if flags_dict.wandb:
        os.environ['WANDB_API_KEY'] = FLAGS.wandb_key
        # os.environ['WANDB_CONFIG_DIR'] = '/home/hlcv_team047/code/'  #for docker environment

    train(flags_dict)




if __name__ == '__main__':
    main()

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from easydict import EasyDict
from texttoimageloader import Text2ImageDataset
from WGAN import Generator, Critic
import os
import wandb
from torchvision.utils import make_grid
import utils
from tensorflow.python.platform import flags
import torchvision.transforms as transforms

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
flags.DEFINE_integer("critic_repeats", 5, 'critic opt / generator opt')
flags.DEFINE_float("lambda1",10.0, "Gradient Penalty Coef")
flags.DEFINE_integer("embed_dim", 256, "text embedding dim")
flags.DEFINE_integer("proj_embed_dim", 256, "projected text embedding dim")

flags.DEFINE_integer("cp_interval", 10, 'checkpoint intervals (epochs)')
flags.DEFINE_integer("log_interval", 10, 'log intervals (steps)')

flags.DEFINE_bool("wandb", False, "Using wandb for logging")
flags.DEFINE_string('wandb_key', '', 'wandb key for logging')


flags.DEFINE_string("pre_trained_critic", '', 'pretrained critic path')
flags.DEFINE_string("pre_trained_generator", '', 'pretrained generator path')
flags.DEFINE_string("experiment_name", 'exp', 'the experiment name')
flags.DEFINE_bool("inter", True, "embedding interpolation")


FLAGS = flags.FLAGS

def grad_penalty(critic, real_img, fake_img, embed, epsilon):
    mixed_img = real_img * epsilon + (1-epsilon) * fake_img
    mixed_score = critic(mixed_img, embed)
    gradient = torch.autograd.grad(inputs = mixed_img,
                                   outputs= mixed_score,
                                   grad_outputs=torch.ones_like(mixed_score),
                                   create_graph= True,
                                   retain_graph= True
                                   )[0]
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1)**2)
    return gp


def train(FLAGS):
    if FLAGS.wandb:
        run = wandb.init(project="Text2Image_CON_WGAN", config=FLAGS, name=FLAGS.experiment_name)

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

    ##init modedls
    generator = torch.nn.DataParallel(Generator(z_dim=FLAGS.z_dim, proj_ebmed_dim=FLAGS.proj_embed_dim, embed_dim=FLAGS.embed_dim).cuda(), range(FLAGS.ngpu))
    critic = torch.nn.DataParallel(Critic(proj_embed_dim=FLAGS.proj_embed_dim, embed_dim=FLAGS.embed_dim ).cuda(), range(FLAGS.ngpu))


    #set optimizers
    C_optimizer = torch.optim.Adam(critic.parameters(), lr= 4*FLAGS.lr, betas=(FLAGS.beta, 0.999))
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta, 0.999))



    iteration = 0

    for epoch in range(FLAGS.num_epochs):
        for data in data_loader:
            iteration += 1
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            hidden = text_encoder.init_hidden(FLAGS.batch_size)

            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            real_image = imags[0].to('cuda')

            # Train the Critic
            crit_loss_all = 0
            for i in range(FLAGS.critic_repeats):
                critic.zero_grad()
                outputs = critic(real_image, sent_emb)
                real_loss = torch.mean(outputs)


                if FLAGS.cls:
                    outputs = critic(real_image[:(FLAGS.batch_size - 1)], sent_emb[1:FLAGS.batch_size])
                    wrong_loss = torch.mean(outputs)

                noise = Variable(torch.randn(real_image.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = generator(noise, sent_emb)

                outputs = critic(fake_images.detach(), sent_emb)
                fake_loss = torch.mean(outputs)

                epsilon = torch.rand(len(real_image), 1, 1, 1, device='cuda', requires_grad=True)

                real_img_inter = (real_image.data).requires_grad_()
                fake_img_inter = (fake_images.data).requires_grad_()
                sent_inter = (sent_emb.data).requires_grad_()

                gp = grad_penalty(critic=critic, real_img= real_img_inter, fake_img= fake_img_inter, embed=sent_inter, epsilon=epsilon)



                if FLAGS.cls:
                    c_loss = 0.5 * (wrong_loss + fake_loss) - real_loss + FLAGS.lambda1 * gp
                else:
                    c_loss = fake_loss - real_loss + FLAGS.lambda1 * gp
                C_optimizer.zero_grad()
                G_optimizer.zero_grad()
                c_loss.backward()
                C_optimizer.step()
                crit_loss_all += (c_loss.item() / FLAGS.critic_repeats)

            # Train the generator
            generator.zero_grad()

            outputs = critic(fake_images, sent_emb)
            g_loss = - outputs.mean()
            C_optimizer.zero_grad()
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
                             "W_dist": fake_loss.item() - real_loss.item(),
                             "Critic Loss": -crit_loss_all,
                             "real_img": _real_img,
                              "fake_img": _fake_img})


        print(epoch)
        if FLAGS.wandb:
            run.log({"Generator Loss_e": g_loss.item(),
                     "Critic Loss_e": - crit_loss_all,
                     "epoch" : epoch})



        if (epoch) % FLAGS.cp_interval == 0:
            utils.save_checkpoint(critic, generator, FLAGS.checkpoints_path, epoch, FLAGS)

def main():
    flags_dict = EasyDict()


    for key in dir(FLAGS):
        flags_dict[key] = getattr(FLAGS, key)
    if flags_dict.wandb:
        os.environ['WANDB_API_KEY'] = "7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87"  #TODO replace with FLAGS.wandb_key
        os.environ['WANDB_CONFIG_DIR'] = '/home/hlcv_team047/code/'

    train(flags_dict)



if __name__ == '__main__':
    main()

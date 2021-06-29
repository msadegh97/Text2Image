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

FLAGS = flags.FLAGS

flags.DEFINE_integer('ngpu', 1, 'number of GPUs')
flags.DEFINE_string('dataset', 'birds', 'name of dataset (birds or flowers)')
flags.DEFINE_integer('z_dim', 100, 'Input noise dimension')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('lr', 0.0002, 'learning rate')
flags.DEFINE_float('beta', 0.5, 'beta')
flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
flags.DEFINE_integer('num_workers', 2, 'number of workers')
flags.DEFINE_bool('cls', True, 'add wrong image loss')
flags.DEFINE_string("checkpoints_path", './models/', 'checkpoints_path')
flags.DEFINE_integer("critic_repeats", 5, 'critic opt / generator opt')
flags.DEFINE_float(" ",10.0, "Gradient Penalty Coef")
flags.DEFINE_integer("embed_dim", 1024, "text embedding dim")
flags.DEFINE_integer("proj_embed_dim", 256, "projected text embedding dim")

flags.DEFINE_integer("cp_interval", 10, 'checkpoint intervals (epochs)')
flags.DEFINE_integer("log_interval", 10, 'log intervals (steps)')

flags.DEFINE_bool("wandb", False, "Using wandb for logging")
flags.DEFINE_string('wandb_key', '', 'wandb key for logging')


flags.DEFINE_string("pre_trained_critic", '', 'pretrained critic path')
flags.DEFINE_string("pre_trained_generator", '', 'pretrained generator path')




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
        run = wandb.init(project="Text2Image_C_WGAN", config=FLAGS)

    dataset_add = '../dataset/'
    ## prepare Data
    if FLAGS.dataset == 'birds':
        dataset = Text2ImageDataset(dataset_add+'birds.hdf5', split=0)  ##TODO split
    elif FLAGS.dataset == 'flowers':
        dataset = Text2ImageDataset(dataset_add+ 'flowers.hdf5', split=0) ##TODO split
    else:
        raise ('Dataset not found')

    data_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    ##init modedls
    generator = torch.nn.DataParallel(Generator(z_dim=FLAGS.z_dim, proj_ebmed_dim=FLAGS.proj_embed_dim, embed_dim=FLAGS.embed_dim).cuda(), range(FLAGS.ngpu))
    critic = torch.nn.DataParallel(Critic(proj_embed_dim=FLAGS.proj_embed_dim, embed_dim=FLAGS.embed_dim ).cuda(), range(FLAGS.ngpu))


    #set optimizers
    C_optimizer = torch.optim.Adam(critic.parameters(), lr= FLAGS.lr, betas=(FLAGS.beta, 0.999))
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta, 0.999))



    iteration = 0

    for epoch in range(FLAGS.num_epochs):
        for sample in data_loader:
            iteration += 1
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            wrong_images = sample['wrong_images']

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()
            wrong_images = Variable(wrong_images.float()).cuda()

            # Train the Critic
            crit_loss_all = 0
            for i in range(FLAGS.critic_repeats):
                critic.zero_grad()
                outputs = critic(right_images, right_embed)
                real_loss = torch.mean(outputs)


                if FLAGS.cls:
                    outputs = critic(wrong_images, right_embed)
                    wrong_loss = torch.mean(outputs)

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = generator(noise, right_embed)

                outputs = critic(fake_images.detach(), right_embed)
                fake_loss = torch.mean(outputs)

                epsilon = torch.rand(len(right_images), 1, 1, 1, device='cuda', requires_grad=True)
                gp = grad_penalty(critic=critic, real_img= right_images, fake_img= fake_images, embed=right_embed, epsilon=epsilon)


                c_loss =  fake_loss - real_loss + FLAGS.lambda1 * gp

                if FLAGS.cls:
                    c_loss = c_loss - (wrong_loss - real_loss)

                c_loss.backward()
                C_optimizer.step()
                crit_loss_all += (c_loss / FLAGS.critic_repeats)
            # Train the generator
            generator.zero_grad()

            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = generator(noise, right_embed)
            outputs = critic(fake_images, right_embed)
            g_loss = -1 * torch.mean(outputs)

            g_loss.backward()
            G_optimizer.step()
            if FLAGS.wandb:

                if iteration % FLAGS.log_interval == 0:

                    run.log({"Generator Loss": g_loss.mean(),
                             "Discriminator Loss": crit_loss_all.mean(),
                             "fake_critic": fake_loss,
                             "real_critic": real_loss})
                    if FLAGS.cls:
                        run.log({"wrong_loss": wrong_loss})

                    real_image_grid = make_grid(right_images, nrow=8, pad_value=1)
                    fake_image_grid = make_grid(fake_images, nrow=8, pad_value=1)
                    _real_img = wandb.Image(real_image_grid, caption="real_images")
                    _fake_img = wandb.Image(fake_image_grid, caption="fake_images")

                    run.log({"real_img": _real_img})
                    run.log({"fake_img": _fake_img})



        if (epoch) % FLAGS.cp_interval == 0:
            utils.save_checkpoint(critic, generator, FLAGS.checkpoints_path, epoch, 'WGAN', FLAGS)

def main():
    flags_dict = EasyDict()


    for key in dir(FLAGS):
        flags_dict[key] = getattr(FLAGS, key)
    if flags_dict.wandb:
        os.environ['WANDB_API_KEY'] = "7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87"  #TODO replace with FLAGS.wandb_key
        os.environ['WANDB_CONFIG_DIR'] = '/home/hlcv_team019/code/'

    train(flags_dict)



if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.optim as optim

### Parameters ######################################################
# Number of channels in the training images. For color images this is 3
#nc = 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

### Models ##########################################################

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class DCGAN:
    def __init__(self, ngpu, device,
                 save_dir=None,
                 lr = 0.0002, beta1 = 0.5):
        ### Models ###############
        self.generator = Generator(ngpu).to(device)
        self.discriminator = Discriminator(ngpu).to(device)

        if (device.type == 'cuda') and (ngpu > 1):
            self.generator = nn.DataParallel(self.generator, list(range(ngpu)))
            self.discriminator = nn.DataParallel(self.discriminator, list(range(ngpu)))

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        ### Optimisers ###########
        # lr: Learning rate for optimizers
        # beta1: Beta1 hyperparam for Adam optimizers
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))

        ### Others ###############
        self.save_dir = save_dir

    def generate(self, noise):
        # Generate fake image batch with G
        return self.generator(noise)


    ### Save and Load ##########################################################
    def save(self):
        torch.save(
            self.generator.state_dict(),
            self.save_dir / f"dcgan_generator.chkpt"
        )
        torch.save(
            self.discriminator.state_dict(),
            self.save_dir / f"dcgan_discriminator.chkpt"
        )

    def load(self, generator_path, discriminator_path):
        generator_state_dict = torch.load(generator_path)
        discriminator_state_dict = torch.load(discriminator_path)

        self.generator.load_state_dict(generator_state_dict)
        self.discriminator.load_state_dict(discriminator_state_dict)

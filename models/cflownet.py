#This code is based on: https://github.com/SimonKohl/probabilistic_unet

from models.unet_blocks import *
from models.unet import Unet
from utils.utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import pdb
from models import flows
from nflib.flows import *
from models.layers import *
from utils.tools import dice_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False,norm=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))
            if i < len(self.num_filters)-1 and norm == True:
                layers.append(nn.BatchNorm2d(output_dim))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, 
            initializers, posterior=False,norm=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, 
                posterior=self.posterior,norm=norm)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding
        #We only want the mean of the resulting hxw image
        encoding = encoding.mean([2,3],True)
        
        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = (self.conv_layer(encoding)).squeeze(-1).squeeze(-1)

        mu, log_sigma = torch.chunk(mu_log_sigma,2,dim=1)
        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return encoding.squeeze(-1).squeeze(-1), dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, 
            no_convs_fcomb, initializers, use_tile=True,norm=False):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'
        if not use_tile:
            self.latent_broadcast = nn.Sequential(
                GatedConvTranspose2d(self.latent_dim, 64, 32, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, self.latent_dim, 5, 1, 2)
            )

        layers = []

        #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
        layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb-2):
            if norm:
                layers.append(nn.BatchNorm2d(self.num_filters[0]))
            layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            
        self.layers = nn.Sequential(*layers)


        self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

        if initializers['w'] == 'orthogonal':
            self.layers.apply(init_weights_orthogonal_normal)
            self.last_layer.apply(init_weights_orthogonal_normal)
        else:
            self.layers.apply(init_weights)
            self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
        else:
            z = z.unsqueeze(2).unsqueeze(2)
            z = self.latent_broadcast(z)
        #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
        feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
        output = self.layers(feature_map)
        return self.last_layer(output)

class glowDensity(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix as
    the base distribution for a sequence of flow based transformations. 
    """
    def __init__(self, num_flows, input_channels, num_filters, no_convs_per_block, 
            latent_dim, initializers, posterior=False,norm=False):
        super(glowDensity, self).__init__()
    
        self.base_density = AxisAlignedConvGaussian(input_channels, num_filters, 
                no_convs_per_block, latent_dim, initializers, posterior=True,norm=norm).to(device)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        self.latent_dim = latent_dim
        # Flow parameters
        self.num_flows = num_flows
        nF_oP = num_flows * latent_dim


        # Normalizing flow layers
        self.norms = [CondActNorm(dim=latent_dim) for _ in range(num_flows)]
        self.InvConvs = [CondInvertible1x1Conv(dim=latent_dim) for i in range(num_flows)]
        self.couplings = [CondAffineHalfFlow(dim=latent_dim,latent_dim=num_filters[-1],
            parity=i%2, nh=4) for i in range(num_flows)] 

        # Amortized flow parameters
        self.amor_W = nn.Sequential(nn.Linear(num_filters[-1], 4),nn.ReLU(),
                nn.Linear(4, num_flows * latent_dim**2),)
        self.amor_s = nn.Linear(num_filters[-1], nF_oP)
        self.amor_t = nn.Linear(num_filters[-1], num_flows)

    def forward(self, input, segm=None):

        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        batch_size = input.shape[0]
        self.ldj = torch.zeros(batch_size).to(device)
        h, z0_density = self.base_density(input,segm)
        z = [z0_density.rsample()]
        W = (self.amor_W(h)).view(batch_size, self.num_flows, self.latent_dim,self.latent_dim)
        s = (self.amor_s(h)).view(batch_size, self.num_flows, self.latent_dim)
        t = self.amor_t(h).view(batch_size, self.num_flows, 1)


        # Normalizing flows
        for k in range(self.num_flows):
            z_k, ldj = self.norms[k](z[k], s[:,k,:], t[:,k,:])
            self.ldj += ldj
            z_k, ldj = self.InvConvs[k](z_k, W[:,k,:,:])
            self.ldj += ldj
            z_k, ldj = self.couplings[k](z_k,h)
            self.ldj += ldj 
            z.append(z_k)

        return self.ldj, z[0], z[-1], z0_density

       
class planarFlowDensity(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix as
    the base distribution for a sequence of flow based transformations. 
    """
    def __init__(self, num_flows, input_channels, num_filters, no_convs_per_block, 
            latent_dim, initializers, posterior=False,norm=False):
        super(planarFlowDensity, self).__init__()
    
        self.base_density = AxisAlignedConvGaussian(input_channels, num_filters, 
                no_convs_per_block, latent_dim, initializers, posterior=True,norm=norm).to(device)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        self.latent_dim = latent_dim
        # Flow parameters
        flow = flows.Planar
        self.num_flows = num_flows
        nF_oP = num_flows * latent_dim
        # Amortized flow parameters
        self.amor_u = nn.Sequential(nn.Linear(num_filters[-1], nF_oP),nn.ReLU(),
                nn.Linear(nF_oP, nF_oP),nn.BatchNorm1d(nF_oP))
        self.amor_w = nn.Sequential(nn.Linear(num_filters[-1], nF_oP),nn.ReLU(),
                nn.Linear(nF_oP, nF_oP),nn.BatchNorm1d(nF_oP))
        self.amor_b = nn.Sequential(nn.Linear(num_filters[-1], num_flows), nn.ReLU(),
            nn.Linear(num_flows, num_flows),nn.BatchNorm1d(num_flows))

        # Normalizing flow layers
        for k in range(num_flows):
            flow_k = flow().to(device)
            self.add_module('flow_' + str(k), flow_k)


    def forward(self, input, segm=None):

        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        batch_size = input.shape[0]
        self.log_det_j = 0.
        h, z0_density = self.base_density(input,segm)
        z = [z0_density.rsample()]

        # return amortized u an w for all flows
        u = self.amor_u(h).view(batch_size, self.num_flows, self.latent_dim, 1)
        w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.latent_dim)
        b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        return self.log_det_j, z[0], z[-1], z0_density

class cFlowNet(nn.Module):
    """
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,256], 
            latent_dim=6, no_convs_fcomb=4, beta=1.0, num_flows=4,norm=False,flow=False,glow=False):

        super(cFlowNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0
        self.flow = flow
        self.flow_steps = num_flows

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, 
                self.initializers, apply_last_layer=False, padding=True, norm=norm).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, 
                self.no_convs_per_block, self.latent_dim,  self.initializers,norm=norm).to(device)

        if flow:
            if glow:
                self.posterior = glowDensity(self.flow_steps, self.input_channels, self.num_filters, self.no_convs_per_block, 
                    self.latent_dim, self.initializers,posterior=True,norm=norm).to(device)
            else:
                self.posterior = planarFlowDensity(self.flow_steps, self.input_channels, self.num_filters, self.no_convs_per_block, 
                    self.latent_dim, self.initializers,posterior=True,norm=norm).to(device)
        else: 
            self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, 
                    self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True,norm=norm).to(device)

        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, 
                self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True,norm=norm).to(device)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
#        pdb.set_trace()
        if training:
            if self.flow:
                self.log_det_j, self.z0, self.z, self.posterior_latent_space = self.posterior.forward(patch, segm)
            else:
                _, self.posterior_latent_space = self.posterior.forward(patch,segm)
                self.z = self.posterior_latent_space.rsample()
                self.z0 = self.z.clone()
        _, self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch,False)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        log_pz = self.prior_latent_space.log_prob(z_prior)
        log_qz = self.posterior_latent_space.log_prob(z_prior)
        return self.fcomb.forward(self.unet_features,z_prior), log_pz, log_qz


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space).sum()
            
        else:
            log_posterior_prob = self.posterior_latent_space.log_prob(self.z)
            log_prior_prob = self.prior_latent_space.log_prob(self.z)
            kl_div = (log_posterior_prob - log_prior_prob).sum()
        if self.flow:
            kl_div = kl_div - self.log_det_j.sum()
        return kl_div

    def elbo(self, segm, mask=None,use_mask = True, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        batch_size = segm.shape[0]
        self.kl = (self.kl_divergence(analytic=analytic_kl, calculate_posterior=False))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, 
                calculate_posterior=False, z_posterior=self.z)
        if use_mask:

            self.reconstruction = self.reconstruction*mask
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return self.reconstruction, self.reconstruction_loss/batch_size, self.kl/batch_size,\
                -(self.reconstruction_loss + self.beta * self.kl)/batch_size

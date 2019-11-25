import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class Discriminator_n_layers(nn.Module):
    def __init__(self, args ):
        super(Discriminator_n_layers, self).__init__()

        n_layers = args.n_D_layers
        in_channels = args.out_channels
        def discriminator_block(in_filters, out_filters, k=4, s=2, p=1, norm=True, sigmoid=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=k, stride=s, padding=p)]
            if norm:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sigmoid:
                layers.append(nn.Sigmoid())
                print('use sigmoid')
            return layers

        sequence = [*discriminator_block(in_channels*2, 64, norm=False)] # (1,64,128,128)

        assert n_layers<=5

        if (n_layers == 1):
            'when n_layers==1, the patch_size is (16x16)'
            out_filters = 64* 2**(n_layers-1)

        elif (1 < n_layers & n_layers<= 4):
            '''
            when n_layers==2, the patch_size is (34x34)
            when n_layers==3, the patch_size is (70x70), this is the size used in the paper
            when n_layers==4, the patch_size is (142x142)
            '''
            for k in range(1,n_layers): # k=1,2,3
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
            out_filters = 64* 2**(n_layers-1)

        elif (n_layers == 5):
            '''
            when n_layers==5, the patch_size is (286x286), lis larger than the img_size(256),
            so this is the whole img condition
            '''
            for k in range(1,4): # k=1,2,3
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
                # k=4
            sequence += [*discriminator_block(2**9, 2**9)] #
            out_filters = 2**9

        num_of_filter = min(2*out_filters, 2**9)

        sequence += [*discriminator_block(out_filters, num_of_filter, k=4, s=1, p=1)]
        sequence += [*discriminator_block(num_of_filter, 1, k=4, s=1, p=1, norm=False, sigmoid=True)]

        self.model = nn.Sequential(*sequence)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        #print("self.model(img_input):  ",self.model(img_input).size())
        return self.model(img_input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, inputA):
        #input = torch.cat((inputA, inputB), 1)
        input  = inputA
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+3):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
        


## for disc loss
class Disc_MultiS_Scale_Loss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(Disc_MultiS_Scale_Loss, self).__init__()
        self.Tensor = tensor
        self.loss = nn.L1Loss()
        
    def __call__(self, disc_inter_feat_list, realB_MultiScale_list):
        indx =0
        n_disc_layer = 5  #cmt
        for i in range(1, len(disc_inter_feat_list)-2):
#             print('len feat list:', len(disc_inter_feat_list))
#             print('gt:real_B: ', realB_MultiScale_list[i].size() )
#             print('inter_feat{}:'.format(i), disc_inter_feat_list[i].size() )
            weight = 2**(5-indx)
            weight = 1/weight
            #print('{} weight: '.format(i), weight)
            if i==1:
                loss = weight*self.loss(realB_MultiScale_list[i], disc_inter_feat_list[i] )
            else:
                loss= loss+ weight*self.loss(realB_MultiScale_list[i], disc_inter_feat_list[i] )
            indx+=1
        return loss

    
    

####################################################
# Initialize discriminator
####################################################
def create_disc_nets(args):
    #discriminator = Discriminator_n_layers(args)
    discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=5, 
                                        norm_layer=nn.BatchNorm2d, 
                                        use_sigmoid=True, getIntermFeat=True)
    
    if torch.cuda.is_available():
        discriminator = discriminator.cuda()

    if args.epoch_start != 0:
        # Load pretrained models
        discriminator.load_state_dict(torch.load('./Exp1_RobotCarReal-deraindrop/saved_models/discriminator_%d.pth' % (args.epoch_start)))
    else:
        # Initialize weights
        discriminator.apply(weights_init_normal)
        print_network(discriminator)

    return  discriminator



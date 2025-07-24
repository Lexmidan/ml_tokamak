"""

Copyright 2022 Vincent Le Guen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Soft
ware"), to deal in the Software without restriction, including withou
t limitation the rights to use, copy, modify, merge, publish, distrib
ute, sublicense, and/or sell copies of the Software, and to permit pe
rsons to whom the Software is furnished to do so, subject to the foll
owing conditions:

The above copyright notice and this permission notice shall be includ
ed in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRE
SS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHAN
TABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM
, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTW
ARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1',nn.GroupNorm( 7 ,F_hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(3,3),
                              padding=(1,1), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]      
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)        # prediction
        next_hidden = hidden_tilde + K * (x-hidden_tilde)   # correction , Haddamard product     
        return next_hidden

class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        
        return self.H , self.H 
    
    def initHidden(self,batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device) )

    def setHidden(self, H):
        self.H = H

        
class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):              
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]          
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size,device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [],[]   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ',i,'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
            self.C.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C
 

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3), stride=stride, padding=1),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=(3,3), stride=stride,padding=1,output_padding=output_padding),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
        
class encoder_E(nn.Module):
    def __init__(self, nc=1, nf=176): #half of image size
        super(encoder_E, self).__init__()
        # input is (1) x 480 x 480
        self.c1 = dcgan_conv(nc, nf, stride=2) # (32) x 250 x 250
        self.c2 = dcgan_conv(nf, nf, stride=1) # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2*nf, stride=2) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)    
        h3 = self.c3(h2)
        return h3

class decoder_D(nn.Module):
    def __init__(self, nc=1, nf=176): #half of image size
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2*nf, nf, stride=2) #(32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=1) #(32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf,out_channels=nc,kernel_size=(3,3),stride=2,padding=1,output_padding=1)  #(nc) x 64 x 64

    def forward(self, input):      
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)  
        return d3  


class encoder_specific(nn.Module):
    def __init__(self, nc=352, nf=352): #image size for both nc and nf
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1) # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)     
        return h2

class decoder_specific(nn.Module):
    def __init__(self, nc=352, nf=352): #image size for both nc and nf
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1) #(64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=1) #(32) x 32 x 32
        
    def forward(self, input):
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)  
        return d2       

        
class EncoderRNN(torch.nn.Module):
    def __init__(self,phycell,convcell, device):
        super(EncoderRNN, self).__init__()
        self.encoder_E = encoder_E()   # general encoder 64x64x1 -> 32x32x32
        self.encoder_Ep = encoder_specific() # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific() 
        self.decoder_Dp = decoder_specific() # specific image decoder 16x16x64 -> 32x32x32 
        self.decoder_Dr = decoder_specific()     
        self.decoder_D = decoder_D()  # general decoder 32x32x32 -> 64x64x1 

        self.encoder_E = self.encoder_E.to(device)
        self.encoder_Ep = self.encoder_Ep.to(device) 
        self.encoder_Er = self.encoder_Er.to(device) 
        self.decoder_Dp = self.decoder_Dp.to(device) 
        self.decoder_Dr = self.decoder_Dr.to(device)               
        self.decoder_D = self.decoder_D.to(device)
        self.phycell = phycell.to(device)
        self.convcell = convcell.to(device)

    def forward(self, input, first_timestep=False, decoding=False):
        input = self.encoder_E(input) # general encoder 1x480x480 -> 480x120x120
    
        if decoding:  # input=None in decoding phase
            input_phys = None
        else:
            input_phys = self.encoder_Ep(input)
        input_conv = self.encoder_Er(input)     

        hidden1, output1 = self.phycell(input_phys, first_timestep)
        hidden2, output2 = self.convcell(input_conv, first_timestep)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])
        
        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp)) # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr   
        output_image = torch.sigmoid( self.decoder_D(concat ))
        return out_phys, hidden1, output_image, out_phys, out_conv        
    

class ClassifierRNN(torch.nn.Module):
    def __init__(self,phycell,convcell, device):
        super(ClassifierRNN, self).__init__()
        self.encoder_E = encoder_E()   # general encoder 64x64x1 -> 32x32x32
        self.encoder_Ep = encoder_specific() # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific() 
        self.decoder_Dp = decoder_specific() # specific image decoder 16x16x64 -> 32x32x32 
        self.decoder_Dr = decoder_specific()     
        self.classifier = CNNClassifier()

        self.encoder_E = self.encoder_E.to(device)
        self.encoder_Ep = self.encoder_Ep.to(device) 
        self.encoder_Er = self.encoder_Er.to(device) 
        self.decoder_Dp = self.decoder_Dp.to(device) 
        self.decoder_Dr = self.decoder_Dr.to(device) 
        self.classifier.to(device)
        self.phycell = phycell.to(device)
        self.convcell = convcell.to(device)

    def forward(self, input, first_timestep=False):
        input = self.encoder_E(input) # general encoder 1x480x480 -> 480x120x120
    
        input_phys = self.encoder_Ep(input)
        input_conv = self.encoder_Er(input)     

        #hidden and output are the same... I don't know what was the autor's intention (visualling the hidden state?)
        hidden1, output1 = self.phycell(input_phys, first_timestep) #list of num_of_phys_layers (1) tensors, each tensor is [batch_size, 8, 88, 88]
        hidden2, output2 = self.convcell(input_conv, first_timestep) #list of num_of_conv_layers (3) tensors, first two tensors are [batch_size, 8, 88, 88] and the last one is [batch_size, 352, 88, 88]

        ######## This is my code. There are many like it, but this one is mine.########
        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])
        #Assuming output2 has len 3, where first 2 tensors of shape 
        # output1_avg = torch.mean(torch.stack(output1), dim=0)
        # output1_avg_avg = output1_avg.mean(dim=1, keepdim=True) # [3,1,88,88]

        # output2_avg = torch.mean(torch.stack(output2[0], output2[1]), dim=0)
        # output2_avg_avg = output2_avg.mean(dim=1, keepdim=True) # [3,1,88,88]

        # output3_avg_avg = output2[2].mean(dim=1, keepdim=True) # [3,1,88,88]

        # image_and_phys = torch.cat((output1_avg_avg, output2_avg_avg, output3_avg_avg), dim=1) #[3,3,88,88]

        classifier_input = decoded_Dp + decoded_Dr # [3, 352, 88, 88]

        output = self.classifier(classifier_input)

        return output
    

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(352, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 22 * 22, 64)  # Adjust size based on the output of the last pooling layer
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(self.bn1(F.leaky_relu(self.conv1(x))))
        x = self.pool(self.bn2(F.leaky_relu(self.conv2(x))))
        x = x.view(-1, 32 * 22 * 22)  # Flatten the tensor for the fully connected layer
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

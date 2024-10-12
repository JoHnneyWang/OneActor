import torch.nn as nn
import torch
import torch.nn.init as init

class AdaLIN(nn.Module):
    def __init__(self, num_features=2048, epsilon=1e-5):
        super(AdaLIN, self).__init__()
        self.epsilon = epsilon
        
        # Scale and shift parameters
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.rho, 0.9)  # Initialize rho to 0.9
        nn.init.constant_(self.beta, 0)    # Initialize beta to 0
        
    def forward(self, x, gamma, beta):  # x [b,n_words,2048], gamma/beta [b, 2048]
        x = x.clone().permute(0, 2, 1)  # [1, 2048, n_words]
        mean = torch.mean(x, dim=2, keepdim=True)
        std = torch.std(x, dim=2, keepdim=True) + self.epsilon
        
        # Instance normalization
        x_norm = (x - mean) / std
        
        # Scale and shift
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)
        x_norm = gamma * x_norm + beta
        
        # Adaptive parameters
        rho = torch.clamp(self.rho, 0, 1)
        beta = torch.clamp(self.beta, -1, 1)
        
        # Layer normalization
        x_mean = torch.mean(x, dim=(1, 2), keepdim=True)
        x_std = torch.std(x, dim=(1, 2), keepdim=True) + self.epsilon
        x_ln = (x - x_mean) / x_std
        
        # Adaptive normalization
        x = rho * x_ln + (1 - rho) * x_norm + beta
        
        return x

class AdaIN(nn.Module):
    def __init__(self, num_features=2048, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        
        # Scale and shift parameters
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.beta, 0)    # Initialize beta to 0
        
    def forward(self, x, gamma, beta):  # x [b,2048], gamma/beta [b, 2048]
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True) + self.epsilon
        
        # Instance normalization
        x_norm = (x - mean) / std

        # Scale and shift
        x_ = gamma * x_norm + beta
        
        return x_
      
class Residual(nn.Module):
  
  def __init__(self,input_channels, num_channels, use_1x1conv=False, strides=1, **kwargs):
    super(Residual, self).__init__(**kwargs)
    self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
    self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    if use_1x1conv:
      self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
    else:
      self.conv3 = None
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)
    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, X):
    
    Y = self.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))
    if self.conv3:
      X = self.conv3(X)
    Y += X
    Y =self.relu(Y)
    return Y

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
  blk = []
  for i in range(num_residuals):
    if i == 0 and not first_block:
      blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
    else:
      blk.append(Residual(num_channels, num_channels))
  return blk

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Projector(nn.Module):
    def __init__(self, in_channels, out_channels,model_type='XL'):
        super(Projector, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(in_channels, in_channels//2, kernel_size=7, stride=2, padding=3),  # /2, /2, /2 -> 640,16,16 / 640,6,6/ 640, 4, 4
                    nn.BatchNorm2d(in_channels//2),
                    nn.ReLU())
        b2=nn.Sequential(*resnet_block(in_channels//2,in_channels//2,2,first_block=True)) # /1, /1, /1  --> 640,16,16 / 640,6,6 / 640, 4, 4
        b3=nn.Sequential(*resnet_block(in_channels//2,in_channels//4,2))  # /2, /2, /2  -> 320,8,8 / 320,3,3 / 320, 2, 2
        b4=nn.Sequential(*resnet_block(in_channels//4,in_channels//8,2))  # /2, /2, /2  -> 160,4,4 / 160,2,2
        if model_type == 'XL':
            self.net=nn.Sequential(b1,b2,b3,b4,Flatten(),nn.Linear(160*4*4, out_channels),nn.LayerNorm(out_channels))
        elif model_type == '2-1':
            self.net=nn.Sequential(b1,b2,b3,b4,Flatten(),nn.Linear(160*2*2, out_channels),nn.LayerNorm(out_channels))
        else:
            self.net=nn.Sequential(b1,b2,b3,Flatten(),nn.Linear(320*2*2, out_channels),nn.LayerNorm(out_channels))
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)
        
        self.net.apply(init_weights)
    
    def forward(self, x): # b, 1280, 32, 32 / b, 1280, 12, 12 / b, 1280, 8, 8
        return self.net(x)


if __name__ == '__main__':
    x = torch.randn(4, 1280, 32, 32)

    blender = Projector(1280, 2048)

    out = blender(x)
    import pdb; pdb.set_trace()

    
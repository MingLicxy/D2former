## Pytorch implementation of Oformer. 
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

#from timm.models.layers import to_2tuple, trunc_normal_
from timm.layers import to_2tuple, trunc_normal_

#TODO 引入基于INR的低通滤波器
#from INR import INR

# 过滤警报
import warnings
warnings.simplefilter("ignore", UserWarning)

################################ 单输入单输出 ################################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

######################## SGFN(DAT) 前馈层 ########################
class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return x1 * x2

class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x
##################################################################

######################## BasicConv(SFNet) ########################
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
##################################################################



######################## Fusion(NeRD) ########################
class Fusion(nn.Module):
    def __init__(self, in_dim=32):
        super(Fusion, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x_q = self.query_conv(x)
        y_k = self.key_conv(y)
        energy = x_q * y_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_y = y * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        y_gamma = self.gamma2(torch.cat((y, attention_y), dim=1))
        y_out = y * y_gamma[:, [0], :, :] + attention_y * y_gamma[:, [1], :, :]

        x_s = x_out + y_out

        return x_s
#############################################################


######################## 交叉融合模块 (TODO 改进点)########################
## SFCA多头注意力机制
class CrossAttention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        # 为每个头设置控制参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        #TODO 交叉注意力中，KV共用投影层，Q采用单独投影层
        # k和v的投影矩阵，通道数X2
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # 深度可分离卷积
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 后处理
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        # k和v来自同一个输入y
        kv = self.kv_dwconv(self.kv(y))
        # k和v按通道维度上连接在一起，处理完成后沿通道维度分开
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        # 变形
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力图
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 后处理
        out = self.project_out(out)
        return out
    
class FusionBlock_att(nn.Module):
    def __init__(self, channels):
        super(FusionBlock_att, self).__init__()
        # 频域和空域的预处理
        self.mam = nn.Conv2d(channels, channels, 3, 1, 1)
        self.cnn = nn.Conv2d(channels, channels, 3, 1, 1)
        # mam->cnn和cnn->mam互融合注意力
        self.mam_att = CrossAttention(dim=channels)
        self.cnn_att = CrossAttention(dim=channels)
        # nn.Sequential()用于将多个层按顺序组合在一起，形成一个"层序列"
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())

    #TODO 两个输入对应两个输出
    def forward(self, mam, cnn):
        #ori = cnn
        mam = self.mam(mam)
        cnn = self.cnn(cnn)
        mam = self.mam_att(mam, cnn)+mam
        cnn = self.cnn_att(cnn, mam)+cnn
        fuse = self.fuse(torch.cat((mam, cnn), 1))
        mam_a, cnn_a = fuse.chunk(2, dim=1)
        cnn = cnn_a * cnn
        mam = mam * mam_a
        #res = mam + cnn
        
        #TODO 替换极端数值
        #res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        mam = torch.nan_to_num(mam, nan=1e-5, posinf=1e-5, neginf=1e-5)
        cnn = torch.nan_to_num(cnn, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return mam, cnn # mam, cnn = my_function()
###################################################################




class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

################################ windows(SwinTransformer) ################################
def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

#TODO 改进点：窗口自注意力
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        #  relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    


#TODO 差分窗口自注意力
# λinit
def lambda_init_s(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

class DiffWindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 window_size, # 窗口尺寸
                 num_heads,
                 qk_bias=True,
                 v_bias=True,
                 qk_scale=None,
                 layer_idx=None, # λinit相关
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.window_size = window_size  
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # scale = 1.0 / math.sqrt(self.head_dim)
        self.scale = qk_scale or self.head_dim**-0.5 # 定义缩放尺度

        #TODO λinit与层索引相关（后续可以尝试直接设置为0.8）
        self.lambda_init = lambda_init_s(layer_idx) 
        #self.lambda_init = 0.8

        #  relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        #TODO 计算空间注意力时投影层用Linear, 可以选择[qkv], [qk,v], [q,k,v]三种投影操作
        self.q1 = nn.Linear(dim, dim, bias=qk_bias)
        self.k1 = nn.Linear(dim, dim, bias=qk_bias)
        self.q2 = nn.Linear(dim, dim, bias=qk_bias)
        self.k2 = nn.Linear(dim, dim, bias=qk_bias)

        self.v = nn.Linear(dim, dim * 2, bias=v_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim * 2, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        # 归一化层采用GroupNorm
        self.subln = nn.LayerNorm(2 * self.head_dim, elementwise_affine=False)

        # Init λ across heads
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)

    def forward(self, x, mask=None):

        b, n, c = x.shape # 输入长度为n的序列    (b, self.num_heads, n, c // self.num_heads)
        q1 = self.q1(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        k1 = self.k1(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        q2 = self.q2(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        k2 = self.k2(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 

        v = self.v(x).reshape(b, n, self.num_heads, 2*self.head_dim).permute(0, 2, 1, 3)
        
        #TODO 分别计算两个注意力图，self.scale放在哪
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        
        # 与移位窗口相关
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        att1 = att1 + relative_position_bias.unsqueeze(0)
        att2 = att2 + relative_position_bias.unsqueeze(0)
        
        # 设置掩码mask
        if mask is not None:
            nw = mask.shape[0]
            att1 = att1.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            att2 = att2.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            att1 = att1.view(-1, self.num_heads, n, n)
            att2 = att2.view(-1, self.num_heads, n, n)

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # Compute λ for each head separately 重参化
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        #TODO 计算差分注意力图
        att = att1 - lambda_full * att2
        att = self.attn_drop(att)

        y = torch.matmul(att, v)  # [b, num_heads, n, 2 * head_dim]
        y = self.subln(y) # 组归一化
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(b, n, 2 * c)
        out = self.proj_drop(self.proj(y))
        return out


## Spatial-wise window-based Transformer block (STB in this paper)
class SpatialTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=8,
                 shift_size=0,
                 mlp_ratio=4.,
                 layer_idx=None,
                 qk_bias=True,
                 v_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)

        #TODO 空间分支采用窗口自注意力
        # self.attn = WindowAttention(
        #     dim,
        #     window_size=to_2tuple(self.window_size),
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop)
        
        #TODO 空间分支采用差分窗口自注意力
        self.attn = DiffWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qk_bias=True,
            v_bias=True,
            qk_scale=qk_scale,
            layer_idx=layer_idx,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def calculate_mask(self, x_size):
        # calculate mask for shift
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        b, c, h, w = x.shape
    
        x = to_3d(x)
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # padding
        size_par = self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - w % size_par) % size_par
        pad_b = (size_par - h % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape
        x_size = (Hd, Wd)

        if min(x_size) == self.window_size:
            self.shift_size = 0
        assert self.window_size <= min(x_size)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        if self.shift_size == 0:
            attn_windows = self.attn(x_windows, mask=None)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, Hd, Wd)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = to_4d(x, h, w)

        return x


# 通道分支主要架构就是照搬Resformer
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

#TODO 改进点：通道注意力
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
#TODO 差分通道自注意力
# λinit
def lambda_init_c(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

class DiffAttention(nn.Module): # layer_idx表示所在层数，影响λ初始化
    def __init__(self, dim, num_heads, bias=False, layer_idx=None):
        super(DiffAttention, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads # 头数
        self.head_size = dim // num_heads # 每头通道数
        
        #TODO 可学习的温度参数，在这里是否采用
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        #TODO λinit与层索引相关
        self.lambda_init = lambda_init_c(layer_idx)
        #self.lambda_init = 0.8

        #TODO 原始投影操作
        #self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        #self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        
        #TODO 这里可以选择[qkv], [qk,v], [q,k,v]三种投影操作
        self.qk = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) # 省略qk的升维卷积
        self.q1_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k1_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.q2_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k2_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        #TODO 
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        #TODO 计算注意力后的组归一化GroupNorm是否需要，归一化需要通道维处于最后
        self.subln = nn.LayerNorm(self.head_size, elementwise_affine=False)

        # Init λ across heads
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)

         

    def forward(self, x):
        b,c,h,w = x.shape

        #qkv = self.qkv_dwconv(self.qkv(x))
        #q,k,v = qkv.chunk(3, dim=1)

        q1 = self.q1_dwconv(self.qk(x))
        k1 = self.k1_dwconv(self.qk(x))
        q2 = self.q2_dwconv(self.qk(x))
        k2 = self.k2_dwconv(self.qk(x))
        v = self.v_dwconv(self.v(x))
        
        #TODO c与(h,w)这两个维度的顺序决定计算的是空间注意力还是通道注意力
        # qk(head c) = (num_heads, head_size) 
        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        #TODO 这个缩放因子一般在空间自注意力中采用
        # scale = 1.0 / math.sqrt(self.head_size)

        # 归一化
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        #TODO 分别计算两个注意力图
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) # cxc
        att2 = torch.matmul(q2, k2.transpose(-2, -1))

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)
        
        # Compute λ for each head separately 重参化
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        #TODO 计算差分注意力图
        att = att1 - lambda_full * att2
        #print("#######################################", att.shape) # [1, 1, 48, 48]
        #print("#######################################", v.shape) # [1, 1, 96, 65536]
        y = torch.matmul(att, v)  # [b, head, c, (h w)]
        
        # 归一化向量的通道维必须位于最后一维
        y = self.subln(y.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) # 组归一化针对Diff计算得到注意力图多样性的特点
        y = y * (1 - self.lambda_init) # 对齐梯度流

        y = rearrange(y, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(y)
        return out


## Channel-wise cross-covariance Transformer block (CTB in this paper)
class ChannelTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor,
                 bias,
                 layer_idx=None,
                 LayerNorm_type=nn.LayerNorm):
        super(ChannelTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        #TODO 通道分支采用普通的多头自注意力
        #self.attn = Attention(dim, num_heads, bias)

        #TODO 通道分支采用差分通道自注意力
        self.attn = DiffAttention(dim, num_heads, bias, layer_idx)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    

#TODO refinement阶段专用块（通道注意力）
class RefineTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor,
                 bias,
                 layer_idx=None,
                 LayerNorm_type=nn.LayerNorm):
        super(RefineTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        #TODO refinement过程采用普通的多头自注意力
        # self.attn = Attention(dim, num_heads, bias)

        #TODO refinement过程采用差分通道自注意力
        self.attn = DiffAttention(dim, num_heads, bias, layer_idx)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


# 浅层特征提取
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False): #BUG 这里的图像处理也涉及图像通道数
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    
# 用于1/2和1/4的浅层特征提取（SFNet）
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(1, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x   

# 用于特征融合（SFNet）
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

   




class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


######################################## OFormer（主体） ########################################
class Difformer(nn.Module): #TODO 注意：模型超参直接在文件中设置
    def __init__(self, 
        inp_channels=1, # 在配置文件中定义
        out_channels=1,
        dim = 48, # 通道数一般48
        num_blocks = [2,4,4], 
        spatial_num_blocks = [2,4,4,6],
        num_refinement_blocks = 4,
        heads = [1,2,4,8], # 注意力头数为什么要递增
        window_size=[16,16,16,16], # 特征图尺度改变，窗口大小不变（用于获取全局感受野）
        drop_path_rate=0.1,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'BiasFree',   ## Other option 'WithBias'
        dual_pixel_task = False 
    ):

        super(Difformer, self).__init__()
        self.alpha = 1
        self.beta = 1

        #########################################TODO  定义浅层特征提取与特征融合(SFNet)  ##################################### 
        self.FAM2 = FAM(dim * 2) # 96
        self.SCM2 = SCM(dim * 2) # 96
        self.FAM4 = FAM(dim * 4) # 192
        self.SCM4 = SCM(dim * 4) # 192
        
        # 多尺度输出层
        self.ConvsOut4 = BasicConv(dim * 4, 1, kernel_size=3, relu=False, stride=1)
        self.ConvsOut2 = BasicConv(dim * 2, 1, kernel_size=3, relu=False, stride=1)
   
        



        #########################################TODO  Bidirectional connection unit (BCU) （双分支交互，可以是改进点）  ##################################### 
        self.Convs = nn.ModuleList()
        self.Convs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3,padding=1,stride=1))
        self.Convs.append(nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2, kernel_size=3,padding=1,stride=1))
        self.Convs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3,padding=1,stride=1))
        self.Convs.append(nn.Conv2d(dim, dim, kernel_size=3,padding=1,stride=1))
        self.DWconvs = nn.ModuleList()
        self.DWconvs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3,padding=1,stride=1,groups=dim * 2))
        self.DWconvs.append(nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2, kernel_size=3,padding=1,stride=1,groups=dim * 2 ** 2))
        self.DWconvs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3,padding=1,stride=1,groups=dim * 2))
        self.DWconvs.append(nn.Conv2d(dim, dim, kernel_size=3,padding=1,stride=1,groups=dim))
        #########################################  end  ##################################### 

        #############################################TODO CrossFusion #############################################
        self.crossfusion1 = FusionBlock_att(dim * 2)
        self.crossfusion2 = FusionBlock_att(dim * 2 ** 2)
        self.crossfusion3 = FusionBlock_att(dim * 2)
        self.crossfusion4 = FusionBlock_att(dim)
        ###########################################################################################################

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(spatial_num_blocks))]  # stochastic depth decay rule
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 


        #TODO 思考如何传入layer_idx
        #####################################  channel-wise branch(Restormer)  ##################################### 
        self.encoder_level1 = nn.Sequential(*[ChannelTransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_idx=i+1, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_idx=i+1, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_idx=i+1, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        #TODO
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias) 
        self.decoder_level3 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_idx=i+1, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        #TODO
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_idx=i+1, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        #TODO
        self.decoder_level1 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_idx=i+1, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
         #########################################  end  ##################################### 


        #TODO 思考如何传入layer_idx
        #####################################  spatial-wise branch(SwinIR)  ##################################### 
        self.encoder1 = nn.Sequential(*[
            SpatialTransformerBlock(dim=dim,
                             num_heads=heads[0], window_size=window_size[0], shift_size=0 if (i % 2 == 0) else window_size[0] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             layer_idx=i+1,
                             drop_path=dpr[sum(spatial_num_blocks[:0]):sum(spatial_num_blocks[:1])][i]
                             ) for i in range(spatial_num_blocks[0])])

        self.d1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder2 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 1),
                             num_heads=heads[1], window_size=window_size[1], shift_size=0 if (i % 2 == 0) else window_size[1] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             layer_idx=i+1,
                             drop_path=dpr[sum(spatial_num_blocks[:1]):sum(spatial_num_blocks[:2])][i]) for i in range(spatial_num_blocks[1])])

        self.d2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder3 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 2),
                             num_heads=heads[2], window_size=window_size[2], shift_size=0 if (i % 2 == 0) else window_size[2] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             layer_idx=i+1,
                             drop_path=dpr[sum(spatial_num_blocks[:2]):sum(spatial_num_blocks[:3])][i]) for i in range(spatial_num_blocks[2])])

        self.d3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        #TODO UNet最底层模块（参数共享）
        self.s_latent = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 3),
                             num_heads=heads[3], window_size=window_size[3], shift_size=0 if (i % 2 == 0) else window_size[3] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             layer_idx=i+1,
                             drop_path=dpr[sum(spatial_num_blocks[:3]):sum(spatial_num_blocks[:4])][i]) for i in range(spatial_num_blocks[3])])

        self.u4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder3 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 2),
                             num_heads=heads[2], window_size=window_size[2], shift_size=0 if (i % 2 == 0) else window_size[2] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             layer_idx=i+1,
                             drop_path=dpr[sum(spatial_num_blocks[:2]):sum(spatial_num_blocks[:3])][i]) for i in range(spatial_num_blocks[2])])

        self.u3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder2 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 1),
                             num_heads=heads[1], window_size=window_size[1], shift_size=0 if (i % 2 == 0) else window_size[1] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             layer_idx=i+1,
                             drop_path=dpr[sum(spatial_num_blocks[:1]):sum(spatial_num_blocks[:2])][i]) for i in range(spatial_num_blocks[1])])

        self.u2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1
        self.reduce1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.decoder1 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim),
                             num_heads=heads[0], window_size=window_size[0], shift_size=0 if (i % 2 == 0) else window_size[0] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             layer_idx=i+1,
                             drop_path=dpr[sum(spatial_num_blocks[:0]):sum(spatial_num_blocks[:1])][i]) for i in range(spatial_num_blocks[0])])
        #####################################  end  ##################################### 


        #####################################TODO  refinement stage(这里为什么专门用CTB做refinement，是Resformer的优良传统)  ##################################### 
        self.refinement = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_idx=i, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):
        # 多尺度输出
        #outputs = list()

        # 多尺度输入
        #inp_img_1 = inp_img
        #inp_img_2 = F.interpolate(inp_img, scale_factor=0.5)
        #inp_img_4 = F.interpolate(inp_img, scale_factor=0.25)

  
        #TODO 浅层特征提取（只在第一层使用）
        #inp_1 = self.patch_embed(inp_img_1)
        #inp_2 = self.SCM2(inp_img_2)
        #inp_4 = self.SCM4(inp_img_4)
        inp = self.patch_embed(inp_img)
        
        ############################## 256x256 ##############################
        out_enc_level1 = self.encoder_level1(inp) # C
        out_enc1 = self.encoder1(inp) # S
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc2 = self.d1_2(out_enc1)

        ##################TODO 多尺度输入在下采样后，交互融合前(1/2)############
        #inp_enc_level2 = self.FAM2(inp_enc_level2, inp_2)
        #inp_enc2 = self.FAM2(inp_enc2, inp_2)


        # 特征交互一
        shortcut = inp_enc_level2 # 暂存
        #TODO 双分支信息交互（潜在改进点）
        inp_enc_level2 = inp_enc_level2 + self.alpha * self.DWconvs[0](inp_enc2) # information fusion
        inp_enc2 = inp_enc2 + self.beta * self.Convs[0](shortcut) # information fusion
        
        # [C,S]->crossfusion->[C,S] 此处的alpha=beta=1
        #inp_enc_level2 = inp_enc_level2 + self.alpha * self.crossfusion1(shortcut, inp_enc2)[0] 
        #inp_enc2 = inp_enc2 + self.beta * self.crossfusion1(shortcut, inp_enc2)[1] 


        ############################## 128x128 ##############################
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc2 = self.encoder2(inp_enc2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc3 = self.d2_3(out_enc2)

        ##################TODO 多尺度输入在下采样后，交互融合前(1/4)############
        #BUG 
        #print('#############################################', inp_enc_level3.shape) # [1, 48*4=192, 80, 45]
        #inp_enc_level3 = self.FAM4(inp_enc_level3, inp_4) 
        #inp_enc3 = self.FAM4(inp_enc3, inp_4)

        # 特征交互二
        shortcut = inp_enc_level3

        inp_enc_level3 = inp_enc_level3 + self.alpha * self.DWconvs[1](inp_enc3) # information fusion
        inp_enc3 = inp_enc3 + self.beta * self.Convs[1](shortcut) # information fusion
        #inp_enc_level3 = inp_enc_level3 + self.alpha * self.crossfusion2(shortcut, inp_enc3)[0] 
        #inp_enc3 = inp_enc3 + self.beta * self.crossfusion2(shortcut, inp_enc3)[1] 


        ############################## 64x64（无特征交互） ##############################
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc3 = self.encoder3(inp_enc3)
        
        #BUG
        inp_enc_level4 = self.down3_4(out_enc_level3) 
        inp_enc4 = self.d3_4(out_enc3)
        
        #TODO 两个UNet最底层采用参数共享的空间Transformer块
        ############################## 32x32 ##############################
        c_latent = self.s_latent(inp_enc_level4) 
        s_latent = self.s_latent(inp_enc4) 

        inp_dec_level3 = self.up4_3(c_latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1) # 残差连接
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) # 利用卷积削减通道维

        inp_dec3 = self.u4_3(s_latent)
        inp_dec3 = torch.cat([inp_dec3, out_enc3], 1)
        inp_dec3 = self.reduce3(inp_dec3)
        


        ############################## 64x64 ##############################
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        out_dec3 = self.decoder3(inp_dec3)

        ##################TODO 多尺度输出在上采样前(1/4)############
        #out_feat_4 = self.FAM4(out_dec_level3, out_dec3)
        #BUG
        #out_4 = self.ConvsOut4(out_feat_4)
        #outputs.append(out_4 + inp_img_4)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        inp_dec2 = self.u3_2(out_dec3)
        inp_dec2 = torch.cat([inp_dec2, out_enc2], 1)
        inp_dec2 = self.reduce2(inp_dec2) # 因为跳跃连接增加了通道数

        # 特征交互三
        shortcut = inp_dec_level2
        inp_dec_level2 = inp_dec_level2 + self.alpha * self.DWconvs[2](inp_dec2) # information fusion
        inp_dec2 = inp_dec2 + self.beta * self.Convs[2](shortcut) # information fusion
        #inp_dec_level2 = inp_dec_level2 + self.alpha * self.crossfusion3(shortcut, inp_dec2)[0] 
        #inp_dec2 = inp_dec2 + self.beta * self.crossfusion3(shortcut, inp_dec2)[1] 

        ############################## 128x128 ##############################
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        out_dec2 = self.decoder2(inp_dec2)

        ##################TODO 多尺度输出在上采样前(1/2)############
        #out_feat_2 = self.FAM2(out_dec_level2, out_dec2)
        #out_2 = self.ConvsOut2(out_feat_2)
        #outputs.append(out_2 + inp_img_2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)

        inp_dec1 = self.u2_1(out_dec2)
        inp_dec1 = torch.cat([inp_dec1, out_enc1], 1)
        inp_dec1 = self.reduce1(inp_dec1)

        # 特征交互四
        shortcut = inp_dec_level1
        inp_dec_level1 = inp_dec_level1 + self.alpha * self.DWconvs[3](inp_dec1) # information fusion
        inp_dec1 = inp_dec1 + self.beta * self.Convs[3](shortcut) # information fusion
        #inp_dec_level1 = inp_dec_level1 + self.alpha * self.crossfusion4(shortcut, inp_dec1)[0] 
        #inp_dec1 = inp_dec1 + self.beta * self.crossfusion4(shortcut, inp_dec1)[1] 

        ############################## 256x256 ##############################
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        out_dec1 = self.decoder1(inp_dec1)
        
        # 最终取双分支输出的cat
        x = torch.cat([out_dec_level1, out_dec1], 1)

        # refinement
        res = self.refinement(x)
        #out_1 = self.output(res)
        #outputs.append(out_1 + inp_img_1)

        if self.dual_pixel_task:
            res = res + self.skip_conv(inp)
            res = self.output(res)
        else:
            res = self.output(res) + inp_img


        #return outputs # 返回具有三个元素的数组
        return res 


# 模型测试
# if __name__== '__main__':
#     #############Test Model Complexity #############
#     from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
#     # x = torch.randn(1, 3, 640, 360)
#     # x = torch.randn(1, 3, 427, 240)
#     x = torch.randn(1, 1, 256, 256)
#     # x = torch.randn(1, 3, 256, 256)

#     model = Difformer()
#     # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
#     print(model)
#     print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
#     print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
#     output = model(x)
#     print(output[0].shape)

# 方法一
if __name__== '__main__':
    #############Test Model Complexity #############
    from thop import profile

    model = Difformer()
    inputs = torch.randn(1, 1, 128, 128)  # 适配你的输入尺寸
    flops, params = profile(model, inputs=(inputs,))
    # 转换单位
    flops_g = flops / 1e9  # GFLOPs
    params_m = params / 1e6  # MParams

    print(f"FLOPs: {flops_g:.2f}G, Params: {params_m:.2f}M")
    
    #############Test Model Running time #############
    model = Difformer().eval().cuda()
    inputs = torch.randn(1, 1, 128, 128).cuda()

    # 预热
    for _ in range(10):
        _ = model(inputs)

    # 创建 CUDA 事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 计时
    start_event.record()
    with torch.no_grad():
        for _ in range(100):
            _ = model(inputs)
    end_event.record()

    # 等待计算完成
    torch.cuda.synchronize()

    # 计算时间（毫秒）
    avg_time = start_event.elapsed_time(end_event) / 100
    print(f"Avg inference time: {avg_time:.2f} ms")    
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange

from timm.models.layers import DropPath

'''
    2d-lift-3d的时序模型，基于Transformer实现
'''


class MLP(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads    # 8头，每个点按8个特征图提取
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape    # [1920, 33, 32]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    # reshape后[1920, 33, 3, 8, 4]，permute后[3, 1920, 8, 33, 4]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) 这时q、k、v分别是[1920, 8, 33, 4]

        # 这里计算方法：softmax(q @ k) @ v
        attn = (q @ k.transpose(-2, -1)) * self.scale    # @在np.array()中相当于np.dot() attn为[1920, 8, 33, 33]，第一个33表示输入33个点，第二个33表示每个点对其他点的贡献值
        attn = attn.softmax(dim=-1)    # [1920, 8, 33, 33]
        attn = self.attn_drop(attn)    # [1920, 8, 33, 33]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)    # 乘以attention矩阵之后 [1920, 33, 32]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PoseTransformer(nn.Module):
    '''
        :param num_frame (int, tuple): input frame number
        :param num_joints (int, tuple): joints number
        :param in_channel (int): number of input channels, 2D joints have 2 channels: (x,y)
        :param embed_dim_ratio (int): embedding dimension ratio
        :param depth (int): depth of transformer
        :param num_heads (int): number of attention heads
        :param mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        :param qkv_bias (bool): enable bias for qkv if True
        :param qk_scale (float): override default qk scale of head_dim ** -0.5 if set
        :param drop_rate (float): dropout rate
        :param attn_drop_rate (float): attention dropout rate
        :param drop_path_rate (float): stochastic depth rate
        :param norm_layer: (nn.Module): normalization layer
    '''
    def __init__(self, num_frame=30, num_joints=33, in_channel=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, out_dim=23 * 3):

        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        # out_dim = num_joints * 3     #### output dimension is num_joints * 3

        # 空间层面：patch的编码及位置编码
        self.Spatial_patch_to_embedding = nn.Linear(in_channel, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        # 时间层面：位置编码
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)    # 位置编码的dropout系数

        # 深度衰减系数
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # 空间层面的块
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # 空间和时间上的norm
        self.spatial_norm = norm_layer(embed_dim_ratio)    # 空间上，自己norm自己的，每个点emb成32dim的
        self.temporal_norm = norm_layer(embed_dim)    # 时间上按照帧数和每一帧关键点的乘积来算

        ####### A easy way to implement weighted mean，使用1d卷积实现加权平均
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )    # 调整之后 [1920, 33, 2]    # 总共b*f帧，每帧33个2d点

        x = self.Spatial_patch_to_embedding(x)    # 做空间特征编码，编码后[1920, 33, 32]    # 每个点编码为32dim的向量
        x += self.Spatial_pos_embed    # 加上位置编码，用nn.Parameter()初始化，默认初始化为0，可训练
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:    # 堆叠4层，反复提取空间层面的特征
            x = blk(x)

        x = self.spatial_norm(x)    # 空间层面的norm
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)    # 调整之后 [64, 30, 33*32]
        return x

    def forward_features(self, x):
        b  = x.shape[0]
        x += self.Temporal_pos_embed    # 加上时间层面的位置编码
        x = self.pos_drop(x)    # [64, 30, 33*32]
        for blk in self.blocks:
            x = blk(x)

        x = self.temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)    # 用Conv1d，降维，30dim-->1dim
        return x


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)    # 提特征之后 [64, 30, 1056]
        x = self.forward_features(x)    # 提特征前 [64, 30, 1056]，提特征后 [64, 1, 69]
        x = self.head(x)    # [64, 1, 69]

        # 这里view成输出的格式
        # x = x.view(b, 1, p, -1)
        x = x.view(b, 1, -1, 3)

        return x

if __name__ == '__main__':
    model = PoseTransformer(num_frame=30, num_joints=33, in_channel=2, embed_dim_ratio=32, depth=4,
                            num_heads=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_path_rate=0, out_dim=23 * 3)
    print(model)

    x = torch.randn([64, 30, 33, 2], dtype=torch.float32)    # [batch_size, num_frame, input_joint_num, input_channel]
    y = model(x)
    print(y.shape)




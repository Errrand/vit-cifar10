'''
定义visual transformer tiny模型
1.define PatchEmbedding
2.Multi-Head Self Attention
3.定义MLP
4.定义Transformer Encoder Block
5.定义Vit-Tiny主模型
'''

import torch
import torch.nn as nn


'''
1.PatchEmbedding=切 Patch + 展平 (Flatten) + Linear
input[128,3,32,32],every patch[3,4,4]
'''
class PatchEmbed(nn.Module):
    def __init__(self,img_size=32,patch_size=4,in_chans=3,embed_dim=192):
        super().__init__()
        self.img_size=img_size
        self.patch_size=patch_size
        self.n_patches=(img_size//patch_size)**2
        #关键，为什么Conv2d相当于
        #卷积核的参数是一组可以学习的权重和偏置，它们和全连接层(Linear)的参数本质上一样，都是在训练过程中不断更新的。其初始值是使用He初始化
        #卷积核的运算：输入的patch与卷积核逐元素相乘（点积），最后再加上偏置。输出一个数。很像y=wx+b
        
        self.proj=nn.Conv2d(  
            in_chans,
            embed_dim,
            patch_size, #卷积核大小等于patch大小
            stride=patch_size
        )
    def forward(self,x):
        #[128,3,32,32]
        x=self.proj(x) #[128,192,8,8]
        x=x.flatten(2) #[128,192,64] [B,C,N],flatten(start_dim,end_dim),flatten(2)为start_dim=2,end_dim=-1
        x=x.transpose(1,2) #[128,64,192] [B,N,C]
        return x
'''
2.Multi-Head Self Attention:token之间信息交互(空间混合)
heads表示有几个注意力头，dim_head表示每个注意力头负责多少维特征，他们满足dim=heads*dim_head
'''
class Attention(nn.Module):
    def __init__(self,dim,heads=3,dropout=0.1): 
        super().__init__()
        #dim:每个token的特征维度
        #heads:注意力头数量
        #dim_head:每个注意力头的维度
        #ex：dim=192,heads=3 ->每个head64维

        self.heads=heads 
        self.dim_head=dim//heads

        #缩放因子，防止点积过大
        self.scale=self.dim_head ** -0.5
        #1.线性映射得到Q,K,V
        self.q_proj=nn.Linear(dim,dim)
        self.k_proj=nn.Linear(dim,dim)
        self.v_proj=nn.Linear(dim,dim)
        #最后的输出投影
        self.out_proj=nn.Linear(dim,dim)
        #Dropout,防止过拟合
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        '''
        输入x的形状:[B,N,D]
        B:batch size N:token数量 D:每个token的特征维度
        '''
        B,N,D=x.shape
        #1.先得到Q/K/V
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        #2.把[B,N,D]拆成多头
        #变成[B,N,heads,dim_head],再转成[B,heads,N,dim_head]
        q=q.reshape(B,N,self.heads,self.dim_head).permute(0,2,1,3)
        k=k.reshape(B,N,self.heads,self.dim_head).permute(0,2,1,3)
        v=v.reshape(B,N,self.heads,self.dim_head).permute(0,2,1,3)
        
        #3.计算注意力分数
        #每个Query和所有Key做点积，得到相似度矩阵,第i个query对第j个key的相似度
        #(Q_len × head_dim) @ (head_dim × K_len)→ (Q_len × K_len)
        attn=torch.matmul(q,k.transpose(-2,-1))*self.scale
        
        #4.做softmax得到注意力权重
        #对每个query token,计算它关注所有key的概率分布，所以必须在K维上做softmax
        attn=torch.softmax(attn,dim=-1)

        #5.dropout防止过拟合
        #随机把一些注意力元素清零
        attn=self.dropout(attn)

        #6.用注意力权重去加权V
        #用注意力权重attn对value做加权求和，得到每个Token的新表示,(N, N)  ×  (N, D)  →  (N, D)
        #attn:为每个token对所有token的关注程度，v:每个token的内容信息
        #attn_probs @ v.即重要的token(权重低)贡献小，不重要的token(权重低)贡献小
        attn=torch.matmul(attn,v)

        #7.重新拼接多个head结果
        #先转回[B,N,heads,dim_head]
        attn=attn.permute(0,2,1,3)
        #展平成[B,N,D]
        attn=attn.reshape(B,N,D)

        #最后做一次输出投影
        out=self.out_proj(attn)
        
        #输出特征再随机失活
        return self.dropout(out) 
'''
MLP:每个token自己做特征增强(通道混合)
输入 (B, N, D)
        ↓
Linear (D → 4D)
        ↓
GELU
        ↓
Dropout
        ↓
Linear (4D → D)
        ↓
Dropout
        ↓
输出 (B, N, D)
'''
class MLP(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.1):
        super().__init__()
        #输入[B,N,D]
        self.fc1=nn.Linear(dim,hidden_dim)
        self.act1=nn.GELU()
        self.fc2=nn.Linear(hidden_dim,dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        x=self.fc1(x)
        x=self.act1(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return self.dropout(x)
'''
一个Transformer Block
Input
  ↓
LayerNorm
  ↓
Multi-Head Self Attention
  ↓
Residual
  ↓
LayerNorm
  ↓
MLP  ← 这里
  ↓
Residual
'''
class Block(nn.Module):
    def __init__(self,dim,heads=3,mlp_ratio=4.0,dropout=0.1):
        super().__init__()
        #输入x[B,N,D]
        self.norm1=nn.LayerNorm(dim)
        self.attn=Attention(dim,heads=heads,dropout=dropout)
        self.norm2=nn.LayerNorm(dim)
        self.mlp=MLP(dim,int(dim*mlp_ratio),dropout=dropout)
    def forward(self,x):
        out=x+self.attn(self.norm1(x))
        out=out+self.mlp(self.norm2(out))
        return out
'''
Input Image:[128,3,32,32]
'''
class Vit_Tiny(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=192,
                 num_classes=10,
                 mlp_ratio=4.0,
                 num_blocks=6,
                 heads=3,
                 dropout=0.1):
        super().__init__()
        #1.Patch and Position Embedding [128,64,192] [B,N,D]
        self.patch_embed=PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim
        )
        self.num_patches=self.patch_embed.n_patches
        # 嵌入CLS Token,[1,1,192]
        #nn.parameter(),告诉pytorch,这个Tensor是模型的可训练参数(需要梯度+被优化器更新)
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))

        
        
        #嵌入Position Embedding,[1,65,192]
        self.pos_embed=nn.Parameter(torch.zeros(1,self.num_patches+1,embed_dim))
        
        #dropout防止过拟合
        self.dropout=nn.Dropout(dropout)

        #2.Encoder Block
        #用nn.ModuleList,pytorch会自动注册所有子层，加入model.parameters(),自动迁移GPU，正常训练
        self.blocks=nn.ModuleList([
            Block(dim=embed_dim,heads=heads,mlp_ratio=mlp_ratio,dropout=0.1)
            for _ in range(num_blocks)
        ])
        
        self.norm=nn.LayerNorm(embed_dim)
        self.head=nn.Linear(embed_dim,num_classes)
    def forward(self,x):
        #x:[128,3,32,32]
        B=x.shape[0]

        #1.patch embedding
        x=self.patch_embed(x) #[B,N,D],B:batch,N:每张图被切成多少个token,D:每个token的特征维度

        #2.加入cls token
        cls_token=self.cls_token.expand(B,-1,-1) #[B,1,D]
        x=torch.cat([cls_token,x],dim=1) #[B,N+1,D]

        #3.加入position embedding
        #矩阵加法需要同型，但是pytorch有广播机制(broadcast),相加时[1,N+1,D]自动扩展成[B,N+1,D]
        x=x+self.pos_embed #[B,N+1,D]

        #4.dropout
        x=self.dropout(x)

        #5.经过多个transformer block
        for block in self.blocks:  #维度不变，特征逐渐增强
            x=block(x) 
        
        #6.取CLS token的输出用于分类
        x=self.norm(x[:,0]) #[B,D]
        x=self.head(x) #[B,10]
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

"Agent Attention: On the Integration of Softmax and Linear Attention"


class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num

        # h=w=window
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)

        # agent_num = pool_size*pool_size
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        # agent bias in attention_1
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))          # block bias (agent_num,h0,w0)  
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))  #  (agent_num,h,1)
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))  #  (agent_num,1,w)


        # agent bias in attention_2
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))          # block bias (agent_num,h0,w0)   
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))  #  (h,1,agent_num)
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))  #  (1,w,agent_num)

        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)


    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        b, n, c = x.shape
        h = int(n ** 0.5)  
        w = int(n ** 0.5)  
        num_heads = self.num_heads 
        head_dim = c // num_heads  

        #  (b,n,c) --qkv-> (b,n,3c) --reshape-> (b,n,3,c) --permute-> (3,b,n,c)
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        #  q:(b,n,c)  k:(b,n,c)  v:(b,n,c)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (b,n,c) --> (b, h, w, c) --> (b,c,h,w)
        q_t = q.reshape(b, h, w, c).permute(0, 3, 1, 2)

        #  (b,c,h,w) --pool-> (b,c, pool_size, pool_size) --reshape-> (b,c,agent_num) --permute-> (b,agent_num,c)
        agent_tokens = self.pool(q_t).reshape(b, c, -1).permute(0, 2, 1)

        # 
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (b,n,c) --> (b,n,h,d) --> (b,h,n,d)   c=h*d
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (b,n,c) --> (b,n,h,d) --> (b,h,n,d)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (b,n,c) --> (b,n,h,d) --> (b,h,n,d)

        # (b,agent_num,c) --reshape-> (b,agent_num,h,d) --permute-> (b,h,agent_num,d)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)


      

        # (h, agent_num, 7, 7) --> (h, agent_num, window, window);   n=window*window, 
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        #(h, agent_num, window, window) --> (1,h,agent_num,window*window) -->repeat--> (b,h,agent_num,n)
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)

        # (1,h,agent_num,window,1) + (1,h,agent_num,1,window) = (1,h,agent_num,window,window) --reshape-> (1,h,agent_num,n) --repeat--> (b,h,agent_num,n)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)

        # (b,h,agent_num,n) + (b,h,agent_num,n) = (b,h,agent_num,n)
        position_bias = position_bias1 + position_bias2

        # (b,h,agent_num,d) @ (b,h,d,n) = (b,h,agent_num,n);  (b,h,agent_num,n) + (b,h,agent_num,n) = (b,h,agent_num,n)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)

        agent_v = agent_attn @ v # (b,h,agent_num,n) @ (b,h,n,d) = (b,h,agent_num,d)



        # (h, agent_num, 7, 7) --> (h, agent_num, window, window)   n=window*window  
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        # (h, agent_num, window, window) --reshape-> (1,h,agent_num,window*window) --permute--> (1,h,window*window,agent_num) --repeat-> (b,h,n,agent_num)
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)

        # (1,h,window,1,agent_num) + (1,h,1,window,agent_num) = (1,h,window,window,agent_num) --reshape--> (1,h,window*window,agent_num) -->repeat--> (b,h,n,agent_num)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)

        # (b,h,n,agent_num) + (b,h,n,agent_num) = (b,h,n,agent_num)
        agent_bias = agent_bias1 + agent_bias2

        # (b,h,n,d) @ (b,h,d,agent_num) = (b,h,n,agent_num);   (b,h,n,agent_num) + (b,h,n,agent_num) = (b,h,n,agent_num)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)

        x = q_attn @ agent_v  # (b,h,n,agent_num)  @ (b,h,agent_num,d) = (b,h,n,d)



        # (b,h,n,d) --> (b,n,h,d) --> (b,n,c);  c=h*d
        x = x.transpose(1, 2).reshape(b, n, c)
        # (b,h,n,d) --transpose--> (b,n,h,d) --reshape--> (b,H,W,c) --permute-> (b,c,H,W);  n=HW
        v_ = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # (b,c,H,W)--permute-->(b,H,W,c)-->reshape-->(b,n,c); (b,n,c)+(b,n,c)=(b,n,c)
        x = x + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x) # (b,n,c)-->(b,n,c)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    # (B, N, C)
    X = torch.randn(1, 196, 768)
    # window = img_size // patch_size = 224 // 16 = 14
    # window*window=token_num  
    # agent_num: 
    Model = AgentAttention(dim=768, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                                   agent_num=49, window=14)
    out = Model(X)
    print(out.shape)
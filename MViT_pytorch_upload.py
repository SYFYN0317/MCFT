import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from scipy.stats import norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class FeedForwardC(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)+x


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask=None):
        # x:[b,n,dim]
        _, _, _, hh = *x.shape, self.heads
        b, n, c = x.shape
        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        # head_num*head_dim = dim
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # split to_qkv(x) from [b, n, inner_dim*3] to [b, n, inner_dim]*3 tuple
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=hh), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)  # [b,head_num,n,n]
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # [b,head_num,n,head_dim]
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out



class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=169, window=13, **kwargs):
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
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 13, 13))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 13, 13))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c

        agent_tokens = self.pool(q[:, :, :].reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v[:, :, :, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x[:, :, :] = x[:, :, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AgentAttention(dim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

        self.mode = mode

    def forward(self, x, mask=None):
        if self.mode == 'MViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        return x

class Attentionc(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, h = x.shape
        n = n/2
        x1 = x[:,:169,:]
        x2 = x[:, 169:, :]

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        # head_num*head_dim = dim
        qkv1 = self.to_qkv(x1).chunk(3, dim = -1) # split to_qkv(x) from [b, n, inner_dim*3] to [b, n, inner_dim]*3 tuple
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv1)
        qkv2 = self.to_qkv(x2).chunk(3, dim = -1) # split to_qkv(x) from [b, n, inner_dim*3] to [b, n, inner_dim]*3 tuple
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv2)
        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots1 = torch.einsum('bhid,bhjd->bhij', q1, k2) * self.scale
        dots2 = torch.einsum('bhid,bhjd->bhij', q2, k1) * self.scale
        # softmax normalization -> attention matrix
        attn1 = dots1.softmax(dim=-1) # [b,head_num,n,n]
        # b2, c2, h12, w12 = attn1.shape
        # z = torch.zeros(size=(b2, c2, h12, w12), dtype=torch.float32).cuda()
        # miu1 = torch.mean(attn1.data)
        # xita1 = torch.std(attn1.data)
        # beta1 = miu1 + xita1
        # attn1 = torch.where(attn1.data >= beta1, attn1, attn1*0.5)
        attn2 = dots2.softmax(dim=-1)  # [b,head_num,n,n]
        # miu2 = torch.mean(attn2.data)
        # xita2 = torch.std(attn2.data)
        # beta2 = miu2 + xita2
        # attn2 = torch.where(attn2.data >= beta2, attn2, attn2*0.5)
        # value * attention matrix -> output
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v2) # [b,head_num,n,head_dim]
        out2 = torch.einsum('bhij,bhjd->bhid', attn2, v1)

        # cat all output -> [b, n, head_num*head_dim]
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out = torch.cat((x1+ out1,x2+out2), dim=1)
        out = self.to_out(out)
        return out
class TransformerC(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

        self.mode = mode

    def forward(self, x, mask=None):
        if self.mode == 'MViT':
            for attnc, ff in self.layers:
                x = attnc(x, mask=mask)
                x = ff(x)
        return x

class TransformerCC(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attentionc(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

        self.mode = mode

    def forward(self, x, mask=None):
        if self.mode == 'MViT':
            for attnc, ff in self.layers:
                x = attnc(x, mask=mask)
                x = ff(x)
        return x




class MViT(nn.Module):
    def __init__(self, patch_size, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=16, dropout=0., emb_dropout=0., mode='MViT'):
        super().__init__()

        nout = 16

        samesize = 1

        self.separable1 = nn.Sequential(
            nn.Conv2d(num_patches[0], num_patches[0], kernel_size=3, padding=samesize, groups=num_patches[0]),
            nn.Conv2d(num_patches[0], nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize, groups=nout),
            nn.Conv2d(nout, nout * 2, kernel_size=1),
            nn.BatchNorm2d(nout * 2),
            nn.GELU(),
            nn.Conv2d(nout * 2, nout * 2, kernel_size=3, padding=samesize, groups=nout * 2),
            nn.Conv2d(nout * 2, nout * 4, kernel_size=1),
            nn.BatchNorm2d(nout * 4),
            nn.GELU()
        )

        self.separable2 = nn.Sequential(
            nn.Conv2d(num_patches[1], num_patches[1], kernel_size=3, padding=samesize, groups=num_patches[1]),
            nn.Conv2d(num_patches[1], nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize, groups=nout),
            nn.Conv2d(nout, nout * 2, kernel_size=1),
            nn.BatchNorm2d(nout * 2),
            nn.GELU(),
            nn.Conv2d(nout * 2, nout * 2, kernel_size=3, padding=samesize, groups=nout * 2),
            nn.Conv2d(nout * 2, nout * 4, kernel_size=1),
            nn.BatchNorm2d(nout * 4),
            nn.GELU()
        )

        grid_size = 1
        vit_patches = (patch_size // grid_size) ** 2 +1
        self.to_patch_embedding2 = nn.Linear(nout * 4, dim)
        self.to_patch_embedding2c = nn.Linear(nout * 4, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, vit_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerCC(dim, depth - 4, heads, dim_head, mlp_dim, dropout, mode)
        # self.transformer3 = TransformerC(dim, depth - 4, heads, dim_head, mlp_dim, dropout, mode)
        self.transformer1 = TransformerC(dim, depth - 2, heads, dim_head, mlp_dim, dropout, mode)
        self.transformer2 = TransformerC(dim, depth - 2, heads, dim_head, mlp_dim, dropout, mode)
        # self.temp = nn.Parameter(torch.ones([]) * 0.07)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.vision_proj_m = nn.Linear(patch_size * patch_size * nout * 4, dim)
        self.text_proj_m = nn.Linear(patch_size * patch_size * nout * 4, dim)
        self.vision_pos_m = nn.Linear(patch_size * patch_size * nout * 8, dim)
        self.text_neg_m = nn.Linear(patch_size * patch_size * nout * 16, dim)
        self.mlp_head0 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.itm_head = nn.Linear(patch_size * patch_size * nout * 8, 2)
        self.pixel_pos_embedding_relative = nn.Parameter(torch.randn(patch_size*patch_size, dim))
        self.alpha = 0.7


    def get_position_embedding(self, x, center_index, cls_token=False):
        center_h, center_w = center_index
        b, n, c = x.shape
        x = x.reshape(b, 15, 15, c).permute(0, 3, 1, 2)
        b, s, h, w = x.shape
        pos_index = []
        for i in range(h):
            temp_index = []
            for j in range(w):
                temp_index.append(max(abs(i - center_h), abs(j - center_w)))
            pos_index.append(temp_index[:])
        pos_index = np.asarray(pos_index)
        pos_index = pos_index.reshape(-1) + 1
        if cls_token:
            pos_index = np.asarray([-1] + list(pos_index))
        pos_emb = self.pixel_pos_embedding_relative[pos_index, :]
        return pos_emb

    def forward(self, x1, x2, label, mask=None):
        # Multimodal Feature Extraction & Tokenization
        x1 = self.separable1(x1)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x1 = self.to_patch_embedding2(x1)  # [b, n, c to dim], n=hw
        b, n, _ = x1.shape
        pos_emb = torch.unsqueeze(self.get_position_embedding(x1, (7, 7), cls_token=False),dim=0)
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x1 = torch.cat((cls_tokens, x1), dim=1)
        # x1 += self.pos_embedding[:, :n]
        x1 += pos_emb

        x1 = self.dropout(x1)

        x2 = self.separable2(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x2 = self.to_patch_embedding2c(x2)  # common subpsace projection: better to be different with self.to_patch_embedding1
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x2 = torch.cat((cls_tokens, x2), dim=1)
        # x2 += self.pos_embedding[:, :n]
        x2 += pos_emb

        x2 = self.dropout(x2)

        # Attention Fusion
        x1 = self.transformer1(x1)
        x2 = self.transformer2(x2)
        bbb, nnn, ccc = x1.shape
        image_features = rearrange(x1, 'b c h -> b (c h)')
        text_features = rearrange(x2, 'b c h -> b (c h)')
        image_features = torch.nn.functional.normalize(self.vision_proj_m(image_features[:, :]), dim=-1)
        text_features = torch.nn.functional.normalize(self.text_proj_m(text_features[:, :]), dim=-1)





        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)



        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()


        sim_targets = torch.zeros(logits_per_image.size()).to(x1.device)
        sim_targets.fill_diagonal_(1)

        sim_i2t_targets = self.alpha * torch.nn.functional.softmax(logits_per_image, dim=1) + (1 - self.alpha) * sim_targets
        sim_t2i_targets = self.alpha * torch.nn.functional.softmax(logits_per_text, dim=1) + (1 - self.alpha) * sim_targets

        loss_i2t = -torch.sum(torch.nn.functional.log_softmax(logits_per_image, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(torch.nn.functional.log_softmax(logits_per_text, dim=1)*sim_t2i_targets,dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2

        # loss_img = torch.nn.functional.cross_entropy(logits_per_image, label.long())
        # loss_text = torch.nn.functional.cross_entropy(logits_per_text, label.long())
        # loss_clip = (loss_img + loss_text) / 2




        x = torch.cat((x1, x2), dim=1)
        x = self.transformer(x)

        # MLP Pre-Head & Head

        output_pos = x
        bs = x1.size(0)
        weights_i2t = torch.nn.functional.softmax(logits_per_image[:, :bs], dim=1)
        weights_t2i = torch.nn.functional.softmax(logits_per_text[:, :bs], dim=1)
        #
        matrix = torch.zeros(bs, bs)
        for i in range(bs):
            matrix[i] =label
        for i in range(bs):
            mask = matrix[i] == matrix[i][i]
            matrix[i][mask] = 0

        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)

        weights_i2t[matrix == 0] = 0
        weights_t2i[matrix == 0] = 0
        #
        #
        # # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item() #t2i改成i2t
            image_embeds_neg.append(x1[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        #
        # # select a negative text for each image
        text_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()  #i2t改成t2i
            text_embeds_neg.append(x2[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        #
        text_embeds_all = torch.cat([x2, text_embeds_neg], dim=0)
        #
        image_embeds_all = torch.cat([image_embeds_neg,x1], dim=0)
        x4 = torch.cat((text_embeds_all, image_embeds_all), dim=1)
        output_neg = self.transformer(x4)
        #
        output_pos = rearrange(output_pos, 'b c h ->b (c h)')
        output_neg = rearrange(output_neg, 'b c h ->b (c h)')
        vl_embeddings = torch.cat([output_pos[:, :], output_neg[:, :]], dim=0)  #这里还有问题
        vl_output = self.itm_head(vl_embeddings)
        #
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                                dim=0).to(x1.device)
        loss_itm = torch.nn.functional.cross_entropy(vl_output, itm_labels)
        # bb,nn,cc = x.shape
        # n1 = (nn // 2) // 2
        # n2 = nn-(nn//2)//2-1
        # xs1 = torch.unsqueeze(x[:, (nn//2)//2, :],dim=1)
        # xs2 = torch.unsqueeze(x[:, nn-(nn//2)//2-1, :],dim=1)
        # xs11 = torch.cat((xs1,xs2),dim=1)
        xs = torch.squeeze(self.mlp_head0(x))  # b-n
        xs = torch.einsum('bn,bnd->bd', xs, x)

        x = self.mlp_head1(xs)
        return x,loss_ita,loss_itm
        # return x,loss_clip,loss_itm

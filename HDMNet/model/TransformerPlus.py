import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer, ConvModule
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import constant_init, normal_init,trunc_normal_init
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from mmseg.ops import resize
from model.MaskMultiheadAttention import MaskMultiHeadAttention


class MixFFN(BaseModule):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MaskMultiHeadAttention(
            in_features=embed_dims, head_num=num_heads, bias=False, activation=None
        )
        torch.nn.MultiheadAttention

    def forward(self, x, hw_shape, source=None, identity=None, mask=None, cross=False):
        x_q = x
        if source is None:
            x_kv = x
        else:
            x_kv = source

        mask_ver = None
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x_kv, hw_shape)
            x_kv = self.sr(x_kv)
            size = x_kv.size()[-2:]
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)

            if mask is not None:
                # mask: (4, 5*60*60, 1)
                # x_kv: (20, 15*15, 64)
                if mask.size(1) != x_kv.size(1):
                    if mask.size(0) != x_kv.size(0):
                        shot = x_kv.size(0) // mask.size(0)
                        mask = rearrange(mask, 'b (n l) c -> (b n) l c', b=mask.size(0), n=shot)  # (bs*shot, h*w, 1)
                    mask_ver = nlc_to_nchw(mask, hw_shape)  # (bs*shot, 1, h, w)
                    mask_ver = F.interpolate(mask_ver, size=size, mode='bilinear', align_corners=True)
                    mask_ver = nchw_to_nlc(mask_ver)  # (bs*shot, h*w, c)

        if identity is None:
            identity = x_q

        out, weight = self.attn(q=x_q, k=x_kv, v=x_kv, mask=mask, cross=cross, mask_ver=mask_ver)
        return identity + self.dropout_layer(self.proj_drop(out)), weight


class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, hw_shape, source=None, mask=None, cross=False):
        if source is None:
            # Self attention
            x, weight = self.attn(self.norm1(x), hw_shape, identity=x, mask=mask)
        else:
            # Cross attention
            x, weight = self.attn(self.norm1(x), hw_shape, source=self.norm1(source), identity=x, mask=mask, cross=cross)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x, weight


class Mlp(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
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
    

class DiscAdaptor(nn.Module):
    def __init__(self, embed_dims, drop=0., mlp_ratio=4):
        super(DiscAdaptor, self).__init__()
        self.embed_dims = embed_dims
        self.drop = drop
        
        self.norm1 = nn.LayerNorm(embed_dims)

        self.q = nn.Linear(embed_dims, embed_dims, bias=False)
        self.k = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v = nn.Linear(embed_dims, embed_dims, bias=False)
        self.proj_x = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.proj_y = nn.Linear(embed_dims * 2, embed_dims, bias=False)

        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn_x = Mlp(in_features=embed_dims, hidden_features=embed_dims * mlp_ratio, act_layer=nn.GELU, drop=drop)
        self.ffn_y = Mlp(in_features=embed_dims, hidden_features=embed_dims * mlp_ratio, act_layer=nn.GELU, drop=drop)

    def Weighted_GAP(self, supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
        return supp_feat

    def forward(self, x, y, shape, mask=None):
        # x: (4, 225, 64)
        # y: (4, 5*225, 64)
        # shape: (15, 15)
        # mask: (4, 5*225, 1)

        b, _, c = x.size()
        h, w = shape
        
        # skip connection
        x_skip = x  # b, n, c
        y_skip = y  # b, n, c
        
        # reshape
        mask = mask.view(b, -1, w, 1).permute(0, 3, 1, 2).contiguous()  # b, 1, shot*h, w
        
        # layer norm
        x = self.norm1(x)
        y = self.norm1(y)
        
        # input projection
        q = self.q(y)  # Support: b, n, c
        k = self.k(x)  # Query: b, n, c
        v = self.v(x)  # Query: b, n, c
        
        # prototype extraction
        q = q.view(b, -1, w, c).permute(0, 3, 1, 2).contiguous()  # b, c, shot*h, w
        fg_pro = self.Weighted_GAP(q, mask).squeeze(-1)  # b, c, 1
        bg_pro = self.Weighted_GAP(q, 1 - mask).squeeze(-1)  # b, c, 1
        
        # normalize
        cosine_eps = 1e-7
        k_norm = torch.norm(k, 2, 2, True)
        fg_pro_norm = torch.norm(fg_pro, 2, 1, True)
        bg_pro_norm = torch.norm(bg_pro, 2, 1, True)
        
        # cosine similarity
        fg_scores = torch.einsum("bmc,bcn->bmn", k, fg_pro) / (torch.einsum("bmc,bcn->bmn", k_norm, fg_pro_norm) + cosine_eps)
        bg_scores = torch.einsum("bmc,bcn->bmn", k, bg_pro) / (torch.einsum("bmc,bcn->bmn", k_norm, bg_pro_norm) + cosine_eps)
        
        # normalization
        fg_scores = fg_scores.squeeze(-1)  # b, n
        bg_scores = bg_scores.squeeze(-1)  # b, n
        
        fg_scores = (fg_scores - fg_scores.min(1)[0].unsqueeze(1)) / (
            fg_scores.max(1)[0].unsqueeze(1) - fg_scores.min(1)[0].unsqueeze(1) + cosine_eps)
        bg_scores = (bg_scores - bg_scores.min(1)[0].unsqueeze(1)) / (
            bg_scores.max(1)[0].unsqueeze(1) - bg_scores.min(1)[0].unsqueeze(1) + cosine_eps)
        
        fg_scores = fg_scores.unsqueeze(-1)
        bg_scores = bg_scores.unsqueeze(-1)
        
        # discriminative region
        scores = fg_scores - bg_scores  # b, n, 1
        pseudo_mask = scores.clone().permute(0, 2, 1).contiguous()
        pseudo_mask = pseudo_mask.view(b, 1, *shape)  # b, 1, h, w
        
        # truncate score
        score_mask = torch.zeros_like(scores)  # b, n, 1
        score_mask[scores < 0] = -100.
        scores = scores + score_mask  # b, n, 1
        
        # softmax
        scores = F.softmax(scores, dim=1)
        
        # output
        query_pro = scores.transpose(-2, -1).contiguous() @ v  # b, 1, c
        
        # similarity-based prototype fusion
        query_pro_norm = torch.norm(query_pro, 2, 2, True)  # b, 1, c
        sim = torch.einsum("bmc,bcn->bmn", query_pro, fg_pro) / (torch.einsum("bmc,bcn->bmn", query_pro_norm, fg_pro_norm) + cosine_eps)
        sim = (sim + 1.) / 2.  # b, 1, 1
        pro = sim * fg_pro.transpose(-2, -1).contiguous() + (1. - sim) * query_pro  # b, 1, c
    
        # projection
        x = x_skip + self.proj_x(torch.cat([x_skip, pro.expand_as(x_skip)], dim=-1))
        y = y_skip + self.proj_y(torch.cat([y_skip, pro.expand_as(y_skip)], dim=-1))

        # ffn    
        x = x + self.ffn_x(self.norm2(x))
        y = y + self.ffn_y(self.norm2(y))
        return x, y, pseudo_mask
    

class MixVisionTransformer(BaseModule):
    def __init__(self,
                 shot=1,
                 in_channels=64,
                 num_similarity_channels = 2,
                 num_down_stages = 3,
                 embed_dims = 64,
                 num_heads = [2, 4, 8],
                 match_dims = 64, 
                 match_nums_heads = 2,
                 down_patch_sizes = [1, 3, 3],
                 down_stridess = [1, 2, 2],
                 down_sr_ratio = [4, 2, 1],
                 mlp_ratio=4,
                 drop_rate=0.1,
                 attn_drop_rate=0.,
                 qkv_bias=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(MixVisionTransformer, self).__init__(init_cfg=init_cfg)
        self.shot = shot

        self.num_similarity_channels = num_similarity_channels
        self.num_down_stages = num_down_stages
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.match_dims = match_dims
        self.match_nums_heads = match_nums_heads
        self.down_patch_sizes = down_patch_sizes
        self.down_stridess = down_stridess
        self.down_sr_ratio = down_sr_ratio
        self.mlp_ratio=mlp_ratio
        self.qkv_bias = qkv_bias

        # ========================================
        # Self attention
        # ========================================
        self.down_sample_layers = ModuleList()
        for i in range(num_down_stages):
            self.down_sample_layers.append(nn.ModuleList([
                PatchEmbed(
                    in_channels=embed_dims,
                    embed_dims=embed_dims,
                    kernel_size=down_patch_sizes[i],
                    stride=down_stridess[i],
                    padding=down_stridess[i] // 2,
                    norm_cfg=norm_cfg),
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=down_sr_ratio[i]),
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=down_sr_ratio[i]),
                build_norm_layer(norm_cfg, embed_dims)[1]
            ]))

        # ========================================
        # Cross attention
        # Modifications:
        # 1. Add ambiguity eliminator before cross attention
        # ========================================
        self.match_layers = ModuleList()
        for i in range(self.num_down_stages):
            level_match_layers = ModuleList([
                DiscAdaptor(self.match_dims, drop=0., mlp_ratio=mlp_ratio),
                TransformerEncoderLayer(
                    embed_dims=self.match_dims,
                    num_heads=self.match_nums_heads,
                    feedforward_channels=self.mlp_ratio * self.match_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=1
                ),
                ConvModule(self.match_dims + self.num_similarity_channels, self.match_dims, kernel_size=3, stride=1, padding=1, norm_cfg=dict(type="SyncBN"))])
            self.match_layers.append(level_match_layers)
        
        # ========================================
        # Output
        # ========================================
        self.parse_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims),
            nn.ReLU()
        ) for _ in range(self.num_down_stages)
        ])
        self.cls = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, 2, kernel_size=1, stride=1, padding=0)
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MixVisionTransformer, self).init_weights()

    def forward(self, q_x, s_x, mask, similarity):
        # q_x: (bs, c, h, w)
        # s_x: (bs*shot, c, h, w)
        # mask: (bs*shot, 1, h, w)
        # similarity: (bs, 1, h, w)

        # ========================================
        # Self attention
        # Modifications:
        # 1. intercept support FG-BG interactions
        # ========================================
        down_query_features = []
        down_support_features = []
        hw_shapes = []
        down_masks = []
        ae_masks = []  # for ambiguity eliminator
        down_similarity = []
        weights = []
        for i, layer in enumerate(self.down_sample_layers):
            # Patch embed
            q_x, q_hw_shape = layer[0](q_x)
            s_x, s_hw_shape = layer[0](s_x)

            # Self attentions
            tmp_mask = resize(mask, s_hw_shape, mode="nearest")
            ae_mask = rearrange(tmp_mask, '(b n) 1 h w -> b (n h w) 1', n=self.shot)
            ae_masks.append(ae_mask)  # for ambiguity eliminator
            q_x, s_x = layer[1](q_x, hw_shape=q_hw_shape)[0], layer[1](s_x, hw_shape=s_hw_shape, mask=ae_mask)[0]
            q_x, s_x = layer[2](q_x, hw_shape=q_hw_shape)[0], layer[2](s_x, hw_shape=s_hw_shape, mask=ae_mask)[0]
            q_x, s_x = layer[3](q_x), layer[3](s_x)
            
            tmp_mask = rearrange(tmp_mask, "(b n) 1 h w -> b 1 (n h w)", n=self.shot)
            tmp_mask = tmp_mask.repeat(1, q_hw_shape[0] * q_hw_shape[1], 1)
            tmp_similarity = resize(similarity, q_hw_shape, mode="bilinear", align_corners=True)
            down_query_features.append(q_x)
            down_support_features.append(rearrange(s_x, "(b n) l c -> b (n l) c", n=self.shot))
            hw_shapes.append(q_hw_shape)
            down_masks.append(tmp_mask)
            down_similarity.append(tmp_similarity)
            if i != self.num_down_stages - 1:
                q_x, s_x = nlc_to_nchw(q_x, q_hw_shape), nlc_to_nchw(s_x, s_hw_shape)

        # ========================================
        # Cross attention
        # Modifications:
        # 1. Add ambiguity eliminator before each cross attention
        # ========================================
        outs = None
        pseudo_masks = []
        for i in range(self.num_down_stages).__reversed__():
            layer = self.match_layers[i]  # 0 - ambiguity eliminator; 1 - cross attention ...

            # ========================================
            # Ambiguity eliminator
            # x: (n, h*w, c)
            # ========================================
            down_query_features[i], down_support_features[i], pseudo_mask = layer[0](
                down_query_features[i], down_support_features[i],
                hw_shapes[i], ae_masks[i]
            )
            pseudo_masks.append(pseudo_mask)

            # ========================================
            # Cross attention
            # ========================================
            out, weight = layer[1](
                x=down_query_features[i], 
                hw_shape=hw_shapes[i], 
                source=down_support_features[i], 
                mask=down_masks[i], 
                cross=True)
            out = nlc_to_nchw(out, hw_shapes[i])
            weight = weight.view(out.shape[0], hw_shapes[i][0], hw_shapes[i][1])
            out = layer[2](torch.cat([out, down_similarity[i]], dim=1))
            weights.append(weight)
            if outs is None:
                outs = self.parse_layers[i](out)
            else:
                outs = resize(outs, size=out.shape[-2:], mode="bilinear")
                outs = outs + self.parse_layers[i](out + outs)
        outs = self.cls(outs)
        return outs, weights, pseudo_masks


class Transformer(nn.Module):
    def __init__(self, shot=1) -> None:
        super().__init__()
        self.shot=shot
        self.mix_transformer = MixVisionTransformer(shot=self.shot)
  
    def forward(self, features, supp_features, mask, similaryty):
        # features: (bs, c, h, w)
        # supp_features: (bs*shot, c, h, w)
        # mask: (bs*shot, 1, h, w)
        # similarity: (bs, 1, h, w)
        shape = features.shape[-2:]
        outs, weights, pseudo_masks = self.mix_transformer(features, supp_features, mask, similaryty)
        return outs, weights, pseudo_masks

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from contrast import l2_normalize, momentum_update, ProjectionHead2D
from sinkhorn import distributed_sinkhorn
from weight_init import trunc_normal_
from ConvModule import DepthwiseSeparableConv, ConvBlock2d, UpBlock
import numpy as np
from decoder import DecoderBranches, DecoderDPS, Decoder2D
import random

def calDist(fts, prototype, scaler=1.):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: N x C x X x Y x Z
        prototype: prototype of one semantic class
            expect shape: 1 x C
    """
    dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
    return dist

class Pretrain(nn.Module):
    def __init__(self, backbone, num_classes=2, in_channels=960, num_prototype=10):
        super(Pretrain, self).__init__()
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.gamma = 0.999
        in_channels = in_channels
        out_channels = 512
        self.momentum = 0.999
        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=self.num_classes)
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.10)
        )
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, out_channels),
                                       requires_grad=True)
        
        self.proj_head = ProjectionHead2D(out_channels, out_channels)
        self.feat_norm = nn.LayerNorm(out_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.dec = DecoderBranches(out_ch=4, chs=[64, 128, 256, 512, 1024])
        # self.dec = Decoder2D(out_ch=4, chs=[64, 128, 256, 512, 1024])
        # self.dec = NoCat(out_ch=4, chs=[64, 128, 256, 512, 1024])
        # self.perturbation = nn.Dropout2d(p=0.5)
        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks, momentum=None):
        IGNORE = 255
        pred_seg = torch.max(out_seg, 1)[1]
        # mask是标签和预测值之间的相同值， masks是特征图和原型之间的相似度计算值
        # print(gt_seg.shape, pred_seg.shape)
        mask = (gt_seg == pred_seg.view(-1))

        # cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        if momentum is None:
            momentum = self.gamma

        sim_list = []
        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)
            # 优化q的分布使其更加符合条件

            m_k = mask[gt_seg == k] # 得到每个像素属于哪个类的掩码

            c_k = _c[gt_seg == k, ...] # 得到一个形状为 (num_pixels_k, channels) 的新张量，其中 num_pixels_k 是类别为 k 的像素数量。

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype
            # 上一行代码：如果这个像素属于类k，则得到其属于每个原型的概率分布值

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            # print(c_k_tile.shape) torch.Size([num_pixels_k, 480])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)
                
                sim = torch.triu(torch.mm(f, f.T), diagonal=1)  # 计算上三角的相似度矩阵
                
                sim_list.append(torch.sum(sim))  # 提取非零元素作为结果

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=momentum)
                protos[k, n != 0, :] = new_value

            # print(k, n.clone().detach().cpu().numpy())

        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)
        return torch.stack(sim_list)

    # def set_m(self, m):
    #     self.momentum = m

    def forward(self, x_, gt_semantic_seg=None, momentum=None, p=False, sim_loss=False, feat=False):
        x = list(self.backbone(x_))
        if p:
            num_l = gt_semantic_seg.shape[0] // 2
            res = random.sample(range(5), 2)
            for i in range(len(x)):
                x_l = x[i][:num_l]
                x_u = x[i][num_l:]
                if i in res:
                    x_p = nn.Dropout2d(0.5)(x_l)
                else:
                    x_p = x_l
                x[i] = torch.cat([x_l, x_p, x_u], dim=0)
            # num_l = gt_semantic_seg.shape[0]
            # res = random.sample(range(5), 2)
            # for i in range(len(x)):
            #     x_l = x[i][:num_l]
            #     x_u = x[i][num_l:]
            #     if i in res:
            #         x_p = nn.Dropout2d(0.5)(x_u)
            #     else:
            #         x_p = x_u
            #     x[i] = torch.cat([x_l, x_u, x_p], dim=0)
        out = self.dec(x)
        _, _, h, w = x[1].size()

        feat1 = x[1]
        feat2 = F.interpolate(x[2], size=(h, w), mode='nearest')
        feat3 = F.interpolate(x[3], size=(h, w), mode='nearest')
        feat4 = F.interpolate(x[4], size=(h, w), mode='nearest')

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        # print(feats.shape)
        c = self.cls_head(feats)

        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: 5*h*w, k: num_class, m: num_prototype, d: dimension
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg, _ = torch.max(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2], w=feats.shape[3])
        
        if feat:
            return _c
            # return masks
        
        if gt_semantic_seg is not None:
            # if p:
            #     gt_semantic_seg = torch.cat([gt_semantic_seg, gt_semantic_seg], dim=0)

            num_l = gt_semantic_seg.shape[0]
            if gt_semantic_seg.ndim == 3:
                gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
            labeled_el = num_l * 112 * 112
            
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest')
            gt_seg = gt_seg.view(-1)
            sim = self.prototype_learning(_c[:labeled_el], out_seg[:num_l], gt_seg, masks[:labeled_el], momentum)
        
        out.append(out_seg)

        if sim_loss:
            return out, torch.mean(sim)
        return out

class PretrainDPS(nn.Module):
    def __init__(self, backbone, num_classes=2, sinkhorn_mode='plain', in_channels=960, num_prototype=10):
        super(PretrainDPS, self).__init__()
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.gamma = 0.999
        in_channels = in_channels
        out_channels = 512
        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=self.num_classes)
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.10)
        )
        self.sinkhorn_mode = sinkhorn_mode
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, out_channels),
                                       requires_grad=True)
        
        self.proj_head = ProjectionHead2D(out_channels, out_channels)
        self.feat_norm = nn.LayerNorm(out_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.dec = DecoderDPS(out_ch=4, chs=[64, 128, 256, 512, 1024])
        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks, momentum=None):
        sinkhorn_mode=self.sinkhorn_mode
        IGNORE = 255
        pred_seg = torch.max(out_seg, 1)[1]
        # mask是标签和预测值之间的相同值， masks是特征图和原型之间的相似度计算值
        # print(gt_seg.shape, pred_seg.shape)
        mask = (gt_seg == pred_seg.view(-1))

        if momentum is None:
            momentum = self.gamma

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q, mode=sinkhorn_mode)
            # 优化q的分布使其更加符合条件

            m_k = mask[gt_seg == k] # 得到每个像素属于哪个类的掩码

            c_k = _c[gt_seg == k, ...] # 得到一个形状为 (num_pixels_k, channels) 的新张量，其中 num_pixels_k 是类别为 k 的像素数量。

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype
            # 上一行代码：如果这个像素属于类k，则得到其属于每个原型的概率分布值

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            # print(c_k_tile.shape) torch.Size([num_pixels_k, 480])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=momentum)
                protos[k, n != 0, :] = new_value

            # print(k, n.clone().detach().cpu().numpy())

        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)


    def forward(self, x_, gt_semantic_seg=None, momentum=None, threshold=0.85, p=False, probs=False):
        x = list(self.backbone(x_))
        if p:
            num_l = gt_semantic_seg.shape[0] // 2
            res = random.sample(range(5), 2)
            for i in range(len(x)):
                x_l = x[i][:num_l]
                x_u = x[i][num_l:]
                if i in res:
                    x_p = nn.Dropout2d(0.5)(x_l)
                else:
                    x_p = x_l
                x[i] = torch.cat([x_l, x_p, x_u], dim=0)
        out = self.dec(x)
        _, _, h, w = x[1].size()

        feat1 = x[1]
        feat2 = F.interpolate(x[2], size=(h, w), mode='nearest')
        feat3 = F.interpolate(x[3], size=(h, w), mode='nearest')
        feat4 = F.interpolate(x[4], size=(h, w), mode='nearest')

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        # print(feats.shape)
        c = self.cls_head(feats)

        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: 5*h*w, k: num_class, m: num_prototype, d: dimension
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg, _ = torch.max(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2], w=feats.shape[3])
            
        if gt_semantic_seg is not None:
            # if p:
            #     gt_semantic_seg = torch.cat([gt_semantic_seg, gt_semantic_seg], dim=0)

            num_l = gt_semantic_seg.shape[0]
            if gt_semantic_seg.ndim == 3:
                gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
            labeled_el = num_l * 112 * 112
            
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest')
            # print(gt_seg.shape, out_seg.shape)
            gt_seg = gt_seg.view(-1)

            self.prototype_learning(_c[:labeled_el], out_seg[:num_l], gt_seg, masks[:labeled_el], momentum)
        
        out.append(out_seg)
        if probs:
            return masks, out
        return out
        
class PretrainPrev(nn.Module):
    def __init__(self, backbone, num_classes=2, sinkhorn_mode='plain', in_channels=960, num_prototype=10):
        super(PretrainPrev, self).__init__()
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.gamma = 0.999
        in_channels = in_channels
        out_channels = 512
        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=self.num_classes)
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.10)
        )
        self.sinkhorn_mode = sinkhorn_mode
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, out_channels),
                                       requires_grad=True)
        
        self.proj_head = ProjectionHead2D(out_channels, out_channels)
        self.feat_norm = nn.LayerNorm(out_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.transposeConv = UpBlock(self.num_classes, self.num_classes)
        self.branchs = nn.ModuleList()
        for i in range(4):
            seq = nn.Sequential(
                ConvBlock2d(self.num_classes, self.num_classes),
                nn.Conv2d(self.num_classes, self.num_classes, 1, padding=0)
            )
            self.branchs.append(seq)
        # self.dec = DecoderBranches(out_ch=4, chs=[64, 128, 256, 512, 1024])
        
        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks, momentum=None):
        sinkhorn_mode=self.sinkhorn_mode
        IGNORE = 255
        pred_seg = torch.max(out_seg, 1)[1]
        # mask是标签和预测值之间的相同值， masks是特征图和原型之间的相似度计算值
        # print(gt_seg.shape, pred_seg.shape)
        mask = (gt_seg == pred_seg.view(-1))

        if momentum is None:
            momentum = self.gamma

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q, mode=sinkhorn_mode)
            # 优化q的分布使其更加符合条件

            m_k = mask[gt_seg == k] # 得到每个像素属于哪个类的掩码

            c_k = _c[gt_seg == k, ...] # 得到一个形状为 (num_pixels_k, channels) 的新张量，其中 num_pixels_k 是类别为 k 的像素数量。

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype
            # 上一行代码：如果这个像素属于类k，则得到其属于每个原型的概率分布值

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            # print(c_k_tile.shape) torch.Size([num_pixels_k, 480])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=momentum)
                protos[k, n != 0, :] = new_value

            # print(k, n.clone().detach().cpu().numpy())

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                    requires_grad=False)

    

    def forward(self, x_, gt_semantic_seg=None, momentum=None, threshold=0.85):
        x = self.backbone(x_)
        _, _, h, w = x[1].size()

        feat1 = x[1]
        feat2 = F.interpolate(x[2], size=(h, w), mode="nearest")
        feat3 = F.interpolate(x[3], size=(h, w), mode="nearest")
        feat4 = F.interpolate(x[4], size=(h, w), mode="nearest")

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        # print(feats.shape)
        c = self.cls_head(feats)

        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: 5*h*w, k: num_class, m: num_prototype, d: dimension
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg, _ = torch.max(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2], w=feats.shape[3])
            
        if gt_semantic_seg is not None:

            if gt_semantic_seg.ndim == 3:
                gt_semantic_seg = gt_semantic_seg.unsqueeze(1)

            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest')
            # print(gt_seg.shape, out_seg.shape)
            gt_seg = gt_seg.view(-1)
            self.prototype_learning(_c, out_seg, gt_seg, masks, momentum)
        

        out = []
        out_seg_ = self.transposeConv(out_seg)
        for branch in self.branchs:
            output = branch(out_seg_)
            # output = F.softmax(output, dim=1)
            out.append(output)
        
        out.append(out_seg)

        # if train:
        return out
        # else:
        #     return (out[0]+out[1]+out[2]+out[3]) / 4

class Extra(nn.Module):
    def __init__(self, backbone, num_classes=4, sinkhorn_mode='plain', in_channels=960, num_prototype=8, extra_num=4):
        super(Extra, self).__init__()
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.gamma = 0.999
        in_channels = in_channels
        out_channels = 512
        self.u_momentum = 0.999
        self.extra_num = extra_num
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.10)
        )
        self.sinkhorn_mode = sinkhorn_mode
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, out_channels), requires_grad=True)
        self.extra_proto = nn.Parameter(torch.zeros(self.num_classes, self.extra_num, out_channels), requires_grad=True)
        self.proj_head = ProjectionHead2D(out_channels, out_channels)
        self.feat_norm = nn.LayerNorm(out_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.dec = DecoderBranches(out_ch=4, chs=[64, 128, 256, 512, 1024])
        # self.perturbation = nn.Dropout2d(p=0.5)
        trunc_normal_(self.prototypes, std=0.02)
        trunc_normal_(self.extra_proto, std=0.02)

    def prototype_learning(self, _c, pred, gt_seg, masks, momentum=None, mode='proto'):
        if momentum is None:
            momentum = self.gamma

        # pred_seg = torch.max(out_seg, 1)[1]

        # mask是标签和预测值之间的相同值， masks是特征图和原型之间的相似度计算值

        mask = (gt_seg == pred)

        # clustering for each class
        if mode == 'proto':
            protos = self.prototypes.data.clone()
        elif mode == 'extra':
            protos = self.extra_proto.data.clone()

        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0 or (init_q.shape[0] < 100 and mode=='extra'):
                continue

            q, indexs = distributed_sinkhorn(init_q)
            # 优化q的分布使其更加符合条件

            m_k = mask[gt_seg == k] # 得到每个像素属于哪个类的掩码

            c_k = _c[gt_seg == k, ...] # 得到一个形状为 (num_pixels_k, channels) 的新张量，其中 num_pixels_k 是类别为 k 的像素数量。

            if mode == 'proto':
                m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            elif mode == 'extra':
                m_k_tile = repeat(m_k, 'n -> n tile', tile=self.extra_num)

            m_q = q * m_k_tile  # n x self.num_prototype
            # 上一行代码：如果这个像素属于类k，则得到其属于每个原型的概率分布值

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=momentum)
                protos[k, n != 0, :] = new_value

        if mode == 'proto':
            self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)
        elif mode == 'extra':
            self.extra_proto = nn.Parameter(l2_normalize(protos), requires_grad=False)

    def set_ulbm(self, m):
        self.u_momentum = m

    def forward(self, x_, gt_semantic_seg=None, momentum=None, p=False, extra=False, edge=None):
        x = list(self.backbone(x_))
        if p:
            num_l = gt_semantic_seg.shape[0] // 2
            res = random.sample(range(5), 2)
            for i in range(len(x)):
                x_l = x[i][:num_l]
                x_u = x[i][num_l:]
                if i in res:
                    x_p = nn.Dropout2d(0.5)(x_l)
                else:
                    x_p = x_l
                x[i] = torch.cat([x_l, x_p, x_u], dim=0)
        out = self.dec(x)
        _, _, h, w = x[1].size()

        feat1 = x[1]
        feat2 = F.interpolate(x[2], size=(h, w), mode='nearest')
        feat3 = F.interpolate(x[3], size=(h, w), mode='nearest')
        feat4 = F.interpolate(x[4], size=(h, w), mode='nearest')

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        # print(feats.shape)
        c = self.cls_head(feats)

        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        if extra:
            self.extra_proto.data.copy_(l2_normalize(self.extra_proto))
            masks_o = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
            
            masks = torch.einsum('nd,kmd->nmk', _c, self.extra_proto)
            masks_all = torch.einsum('nd,kmd->nmk', _c, torch.cat([self.prototypes, self.extra_proto], dim=1))
            out_seg, _ = torch.max(masks_all, dim=1)
        else:
            self.prototypes.data.copy_(l2_normalize(self.prototypes))
            masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
            out_seg, _ = torch.max(masks, dim=1)

        out_seg = self.mask_norm(out_seg)
        pred = torch.argmax(out_seg, dim=1)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2], w=feats.shape[3])
            
        if gt_semantic_seg is not None:
            # if p:
            #     gt_semantic_seg = torch.cat([gt_semantic_seg, gt_semantic_seg], dim=0)

            num_l = gt_semantic_seg.shape[0]
            if gt_semantic_seg.ndim == 3:
                gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
            labeled_el = num_l * 112 * 112
            
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest')
            gt_seg = gt_seg.view(-1)
            if extra:
                conflict_idx = pred[:labeled_el] != gt_seg
                self.prototype_learning(_c[:labeled_el][conflict_idx], pred[:labeled_el][conflict_idx], gt_seg[conflict_idx], masks[:labeled_el][conflict_idx], momentum=0.8, mode='extra')
                if edge is not None:
                    edge = F.interpolate(edge.float(), size=feats.size()[2:], mode='nearest')
                    out_seg_temp = out_seg.detach().clone()[num_l:]
                    out_seg_temp[:, 0:1][edge==1] = 0
                    pred_seg = torch.argmax(out_seg_temp, dim=1).flatten()
                    conflict_u_idx = pred_seg != pred[labeled_el:]
                    self.prototype_learning(_c[labeled_el:][conflict_u_idx], pred_seg[conflict_u_idx], pred_seg[conflict_u_idx], masks[labeled_el:][conflict_u_idx], momentum=0.9, mode='extra')
            else:
                self.prototype_learning(_c[:labeled_el], pred[:labeled_el], gt_seg, masks[:labeled_el], momentum)
                if edge is not None:
                    edge = F.interpolate(edge.float(), size=feats.size()[2:], mode='nearest')
                    out_seg_temp = out_seg.detach().clone()[num_l:]
                    out_seg_temp[:, 0:1][edge==1] = 0
                    pred_seg = torch.argmax(out_seg_temp, dim=1).flatten()
                    self.prototype_learning(_c[labeled_el:], pred_seg, pred_seg, masks[labeled_el:], momentum=self.u_momentum)
        
        out.append(out_seg)

        return out
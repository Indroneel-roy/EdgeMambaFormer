import torch
import torch.nn.functional as F
import timm
import torch.nn as nn

class BasicConv2d(nn.Module):
    """F(.) of Sec. III-C: 3x3 conv (padding 1) + BatchNorm + ReLU."""
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__()
        self.conv=nn.Conv2d(in_c,out_c,k,padding=p,bias=False)
        self.bn=nn.BatchNorm2d(out_c); self.relu=nn.ReLU(inplace=True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))


class CFM(nn.Module):
    """Cascaded Fusion Module. Eq. 1-2; Table III 'CFM' (8 BasicConv2d)."""
    def __init__(self, ch=32):
        super().__init__()
        self.F1=BasicConv2d(ch,ch);        self.F2=BasicConv2d(ch,ch)       # [32,32,3,1]
        self.F3=BasicConv2d(2*ch,2*ch)                                       # [64,64,3,1]
        self.F4=BasicConv2d(ch,ch);        self.F5=BasicConv2d(ch,ch)       # [32,32,3,1]
        self.F6=BasicConv2d(2*ch,2*ch)                                       # [64,64,3,1]
        self.F7=BasicConv2d(3*ch,3*ch)                                       # [96,96,3,1]
        self.F8=BasicConv2d(3*ch,ch)                                         # [96,32,3,1]

    @staticmethod
    def _up(t,ref):
        return F.interpolate(t,size=ref.shape[-2:],mode='bilinear',align_corners=False)

    def forward(self,x2,x3,x4):
        # Eq.1  X34 = F3( Concat( F1(X4^) (*) X3', F2(X4^) ) )      [up x2]
        x4_3 = self._up(x4,x3)
        X34  = self.F3(torch.cat([self.F1(x4_3)*x3, self.F2(x4_3)],1))
        # Eq.2  T1 = F8(F7( Concat( F4(X4^) (*) F5(X3^) (*) X2', F6(X34^) ) ))  [up x4, x2, x2]
        x4_2, x3_2, X34_2 = self._up(x4,x2), self._up(x3,x2), self._up(X34,x2)
        return self.F8(self.F7(torch.cat(
            [self.F4(x4_2)*self.F5(x3_2)*x2, self.F6(X34_2)],1)))


class ChannelAttention(nn.Module):
    """Eq.4. H1/H2 share parameters; 1x1 conv reduces C by 16x, ReLU, 1x1 back.
    Table III: AvgPool2d[1], AvgPool2d[1], Conv2d[64,4,1,0], ReLU,
    Conv2d[4,64,1,0], Sigmoid."""
    def __init__(self, c, r=16):
        super().__init__()
        self.avg=nn.AdaptiveAvgPool2d(1); self.mx=nn.AdaptiveMaxPool2d(1)
        self.H=nn.Sequential(nn.Conv2d(c,c//r,1,bias=False),nn.ReLU(inplace=True),
                             nn.Conv2d(c//r,c,1,bias=False))
    def forward(self,x):
        return torch.sigmoid(self.H(self.mx(x))+self.H(self.avg(x)))*x


class SpatialAttention(nn.Module):
    """Eq.5. G(.) is a 7x7 conv with padding 3. Table III: Conv2d[2,1,7,3]."""
    def __init__(self,k=7):
        super().__init__(); self.G=nn.Conv2d(2,1,k,padding=k//2,bias=False)
    def forward(self,x):
        a=torch.cat([x.max(1,keepdim=True)[0], x.mean(1,keepdim=True)],1)
        return torch.sigmoid(self.G(a))*x


class CIM(nn.Module):
    """Camouflage Identification Module. Eq.3: T2 = Att_s(Att_c(X1))."""
    def __init__(self,c=64):
        super().__init__(); self.ca=ChannelAttention(c); self.sa=SpatialAttention()
    def forward(self,x1): return self.sa(self.ca(x1))


class GCN(nn.Module):
    """Graph convolutional layer, Table III GCN[num_state=16, num_node=16].
    Node interaction along the node axis with a residual, then a state update."""
    def __init__(self, num_state=16, num_node=16):
        super().__init__()
        self.node=nn.Conv1d(num_node,num_node,1)
        self.state=nn.Conv1d(num_state,num_state,1,bias=False)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):                                   # (B, state, node)
        h=self.node(x.permute(0,2,1)).permute(0,2,1)
        return self.state(self.relu(h+x))


class SAM(nn.Module):
    """Similarity Aggregation Module. Eq.6-10; Table III 'SAM'.
    AvgPool2d[6] then a center crop gives V in R^{4x4x16}."""
    def __init__(self, ch=32, mid=16, c_low=64, pool=6, crop=4):
        super().__init__()
        self.Wtheta=nn.Conv2d(ch,mid,1)                     # Eq.6  [32,16,1]
        self.Wphi  =nn.Conv2d(ch,mid,1)                     # Eq.6  [32,16,1]
        self.Wg    =BasicConv2d(c_low,ch,k=1,p=0)           # [64,32,1,0]
        self.pool  =nn.AdaptiveAvgPool2d(pool)
        self.gcn   =GCN(mid,crop*crop)                      # GCN[16,16]
        self.Wz    =nn.Conv2d(mid,ch,1)                     # [16,32,1]

    def forward(self,T1,T2):
        B=T1.shape[0]; h,w=T1.shape[-2:]
        Q=self.Wtheta(T1); K=self.Wphi(T1)                                  # Eq.6
        # F(.): reduce T2 to 32 ch, align to T1, softmax over channels, 2nd channel
        g=F.interpolate(self.Wg(T2),size=(h,w),mode='bilinear',align_corners=False)
        T2p=torch.softmax(g,dim=1)[:,1:2]                                   # (B,1,h,w)
        V=self.pool(K*T2p)[:,:,1:-1,1:-1]                                   # Eq.7 -> (B,16,4,4)

        Vf=V.flatten(2); Kf=K.flatten(2); Qf=Q.flatten(2)
        f=torch.softmax(torch.bmm(Vf.transpose(1,2),Kf),dim=-1)             # Eq.8 (B,node,hw)
        nodes=torch.bmm(Qf,f.transpose(1,2))                                # (B,state,node)
        Gr=self.gcn(nodes)                                                  # (B,16,16)
        Yp=torch.bmm(Gr,f).view(B,-1,h,w)                                   # Eq.9 (B,16,h,w)
        return T1+self.Wz(Yp)                                               # Eq.10


class PolypPVT(nn.Module):
    """Sec. III-A. Prediction is P1 + P2."""
    def __init__(self, ch=32, pretrained=True):
        super().__init__()
        self.backbone=timm.create_model('pvt_v2_b2',pretrained=pretrained,
                                        features_only=True,out_indices=(0,1,2,3))
        c1,c2,c3,c4=self.backbone.feature_info.channels()          # 64,128,320,512
        # "adjust the channel of X2, X3, X4 to 32 through three convolutional units"
        self.t2=BasicConv2d(c2,ch); self.t3=BasicConv2d(c3,ch); self.t4=BasicConv2d(c4,ch)
        self.cfm=CFM(ch); self.cim=CIM(c1); self.sam=SAM(ch,16,c1)
        self.out_CFM=nn.Conv2d(ch,1,1)                             # P1 head
        self.out_SAM=nn.Conv2d(ch,1,1)                             # P2 head

    def forward(self,x):
        size=x.shape[-2:]
        X1,X2,X3,X4=self.backbone(x)
        T1=self.cfm(self.t2(X2),self.t3(X3),self.t4(X4))           # H/8  x 32
        T2=self.cim(X1)                                            # H/4  x 64
        Z =self.sam(T1,T2)                                         # H/8  x 32
        P1=F.interpolate(self.out_CFM(T1),size=size,mode='bilinear',align_corners=False)
        P2=F.interpolate(self.out_SAM(Z), size=size,mode='bilinear',align_corners=False)
        return P1,P2

print('model defined')
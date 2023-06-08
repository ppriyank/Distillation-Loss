import torch
from torch.nn import functional as F
import torch.nn as nn 

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from einops import rearrange, repeat
import matplotlib.pyplot as plt


class DDL(torch.nn.Module):
    """ Implented of  "https://arxiv.org/abs/2002.03662"
    # https://github.com/Tencent/TFace/blob/master/recognition/torchkit/loss/ddl.py
    """
    def __init__(self, pos_kl_weight=0.1, neg_kl_weight=0.02,
                 order_loss_weight=0.5, positive_threshold=0.0):
        super(DDL, self).__init__()
        self.pos_kl_weight = pos_kl_weight
        self.neg_kl_weight = neg_kl_weight
        self.order_loss_weight = order_loss_weight
        self.positive_threshold = positive_threshold
        
        delta = 2/ (1000)   
        self.register_buffer('t', torch.arange(-1, 1.0, delta).view(-1, 1).t())

    def plot_histogram(self, a, b, c, d, name=""): 
        bins = a.shape[-1]

        x_axis = range(bins)
        for e in [a,b,c,d]:
            if e is not None:
                plt.bar(x_axis, e.view(-1).cpu().detach().numpy(), align='center')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.savefig(name + ".png")
        plt.clf()

    def calculate_distribution(self, x,  label_grid):
        x = F.normalize(x, p=2, dim=-1).squeeze(1)
        Cos_Dist = torch.mm(x, x.transpose(0, 1))
        # Cos_Dist = torch.triu(Cos_Dist, diagonal=1)

        negative_labels = (1 - label_grid)
        positive_labels = label_grid * (1 - torch.eye(label_grid.shape[0]).cuda())
        # positive_labels = label_grid

        dists = Cos_Dist.view(-1, 1)
        DELTA = torch.mm(dists, torch.ones_like(self.t)) - torch.mm(torch.ones_like(dists), self.t)
        DELTA = torch.exp(-0.5 * torch.pow((DELTA / 0.1), 2))

        neg_p = self._histogram2(DELTA, negative_labels)
        # self.plot_histogram(a=neg_p, b=None, c=None, d=None, name="test")
        
        pos_p = self._histogram2(DELTA, positive_labels)
        # self.plot_histogram(a=neg_p, b=pos_p, c=None, d=None, name="test2")
        # neg_p = self._histogram(neg_distance)

        pos_distance = Cos_Dist * positive_labels
        E_pos = pos_distance.sum(-1) / positive_labels.sum(-1)
        # pos_distance, _ = torch.min(pos_distance, dim=-1)

        neg_distance = Cos_Dist * negative_labels
        E_neg = neg_distance.sum(-1) / negative_labels.sum(-1)
        # neg_distance, _ = torch.max(neg_distance, dim=-1)
        
        return neg_p , pos_p, E_neg, E_pos

    def forward(self, T_F, S_F, labels): 

        labels = labels.argmax(-1)
        label_grid = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_grid = label_grid.float()

        neg_teacher , pos_teacher, E_neg_teacher, E_pos_teacher = self.calculate_distribution(T_F,  label_grid)
        # self.plot_histogram(x=neg_teacher, y=pos_teacher, name="Teacher_his")
        neg_student , pos_student, E_neg_student, E_pos_student = self.calculate_distribution(S_F,  label_grid)        
        # self.plot_histogram(a=neg_student, b=pos_student, c=neg_teacher, d=pos_teacher, name="Histogram")

        pos_kl = self.pos_kl_weight * self._kl(pos_teacher, pos_student)
        neg_kl = self.neg_kl_weight * self._kl(neg_teacher, neg_student)

        o1 = torch.abs(E_pos_teacher - E_pos_student).mean()
        o2 = torch.abs(E_neg_teacher - E_neg_student).mean()
        o3 = (E_pos_student - E_neg_student).mean()
        # order_loss = E_pos_teacher.unsqueeze(0) - E_neg_student.unsqueeze(1)
        # order_loss = order_loss.sum()
        order_loss = o1 + o2 + o3
        order_loss = self.order_loss_weight * order_loss
        
        ddl_loss = pos_kl + neg_kl + order_loss
        return ddl_loss

    def _kl(self, anchor_distribution, distribution):
        loss = F.kl_div(torch.log(distribution + 1e-9), anchor_distribution + 1e-9, reduction="batchmean")
        return loss

    def _histogram2(self, DELTA, labels ):
        labels = labels.view(-1,1)
        DELTA_L =   DELTA * labels      
        DELTA_L = torch.sum(DELTA_L, 0, keepdim=True)
        # DELTA_L = torch.sum(DELTA_L, 1)
        DELTA_L_normed = DELTA_L / labels.sum()
        return DELTA_L_normed
        
    def _histogram(self, dists):
        dists = dists.view(-1, 1)
        simi_p = torch.mm(dists, torch.ones_like(self.t)) - torch.mm(torch.ones_like(dists), self.t)
        simi_p = torch.sum(torch.exp(-0.5 * torch.pow((simi_p / 0.1), 2)), 0, keepdim=True)
        p_sum = torch.sum(simi_p, 1)
        simi_p_normed = simi_p / p_sum
        return simi_p_normed

def mse_loss(x,y, **misc):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    diff = x - y 
    diff = (diff ** 2).sum(-1)
    diff = diff.mean()
    return diff

def l1_loss(x,y, **misc):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    diff = x - y 
    diff = diff.abs().sum(-1)
    diff = diff.mean()
    return diff

def mkmmd(x, y, factors=[1,2,4,8,16], **misc):
    # x == teacher
    # y == student
    x = F.normalize(x, p=2, dim=-1).squeeze()
    y = F.normalize(y, p=2, dim=-1).squeeze()
    
    dist_x_x = cuda_eucledian_dist(x,x)
    dist_y_y = cuda_eucledian_dist(y,y)
    dist_x_y = cuda_eucledian_dist(x,y)

    N = dist_x_x.shape[0]
    hint_loss = 0  

    sigma_x_x_2 = (dist_x_x).mean()  
    sigma_y_y_2 = (dist_y_y).mean()  
    sigma_x_y_2 = (dist_x_y).mean() 
    
    for sigma_factor in factors:
        K_X_X = ((-dist_x_x) / (2 * sigma_factor * sigma_x_x_2)).exp() 
        K_Y_Y = ((-dist_y_y) / (2 * sigma_factor * sigma_y_y_2)).exp() 
        K_X_Y = ((-dist_x_y) / (2 * sigma_factor * sigma_x_y_2)).exp() 
        temp_hint_loss = K_X_X.sum() + K_Y_Y.sum() - 2 * K_X_Y.sum() 
        hint_loss += temp_hint_loss / (N ** 2)

    return hint_loss

class Soft_Entropy:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9137263
    # https://ieeexplore.ieee.org/document/9098036
    def __init__(self, lambda1=0, lambda2=0, tau=0, mode="None"):
        self.base_criterion = SoftTargetCrossEntropy()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tau = tau
        self.mode = mode

    def __call__(self, teacher_logits, student_logits, student_logits_teacher_features, **misc):
        entropy1, entropy2 = 0, 0 
        teacher_logits = teacher_logits / self.tau
        if self.mode == "student_softening":
            student_logits = student_logits / self.tau
            
        if self.lambda1:
            # q_pre = (teacher_logits).softmax(-1)
            # p_k = student_logits.softmax(-1)
            # entropy1 = -(q_pre *  p_k.log()).sum(-1).mean()
            entropy1 = self.base_criterion(student_logits, teacher_logits.softmax(-1))
            entropy1 = entropy1 * self.lambda1
            
        if self.lambda2:
            # p_k_pre = student_logits_teacher_features.softmax(-1)        
            # entropy2 = -(q_pre *  p_k_pre.log()).sum(-1).mean()
            entropy2 = self.base_criterion(student_logits_teacher_features, teacher_logits.softmax(-1))
            entropy2 = entropy2 * self.lambda2

        return entropy1 + entropy2 

class Base_DistillationLoss(torch.nn.Module):
    def __init__(self, multiple_lrs=None, spatial_avg=None, tau=3,  
        features_index= [-1], lambda1= 1.0, lambda2= 1.0, lambda3= 1.0, mode=None, alpha=1.0, beta=1.0, 
        gamma=1.0, **kwargs):
        super().__init__()
        
        self.tau = tau
        
        self.alpha = alpha 
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        

        self.base_criterion = SoftTargetCrossEntropy()

        self.features_index = features_index
        
        if spatial_avg == "mean":
            self.spatial_avg = lambda x: x.mean(1).unsqueeze(1) if len(x.shape) > 2 else x.unsqueeze(1)
        else:
            self.spatial_avg = lambda x: x
        
        self.multiple_lrs = multiple_lrs
        if self.multiple_lrs:
            if spatial_avg == "mean":
                self.spatial_avg_lr = lambda x: x.mean(2).unsqueeze(2) if len(x.shape) > 3 else x.unsqueeze(2)
            else:
                self.spatial_avg_lr = lambda x: x

        self.mode = mode

    def forward(self, teacher_logits, teacher_features, student_logits, student_features, samples_student, targets):
        '''
        do nothing
        '''
        return loss

class DistillationLoss(Base_DistillationLoss):
    def __init__(self, aux_loss=None, separate_centers=True, logit_loss=None, **args):
        super().__init__(**args)
        
        self.aux_loss = None
        self.logit_loss = None

        if aux_loss == "DDL":
            self.aux_loss = DDL().cuda()
        elif aux_loss == "l2":
            self.aux_loss = mse_loss
        elif aux_loss == "l1":
            self.aux_loss = l1_loss
        elif aux_loss == "Degree_Order_Loss":
            self.aux_loss = Degree_Order_Loss(
                mode=self.mode, lambda1=self.lambda1, 
                lambda2=self.lambda2, lambda3=self.lambda3,
                separate_centers=separate_centers )
        elif aux_loss == "MLRL":
            self.aux_loss = MLRL(mode=self.mode, lambda1=self.lambda1, lambda2=self.lambda2, lambda3=self.lambda3)
        elif aux_loss == "MK-MMD":
            self.aux_loss = mkmmd

        if logit_loss == "soft_entropy":
            self.logit_loss = Soft_Entropy(lambda1=self.lambda1, lambda2=self.lambda2, tau=self.tau, mode=self.mode)

        print(f"""
        Entropy Loss weight {self.alpha}, 
        Aux loss weight {self.beta}, 
        Logit loss {self.gamma}
        """)

    def forward(self, teacher_logits, teacher_features, student_logits, student_features, labels, student_logits_teacher_features=None, **kwargs):
        '''
        teacher_logits == (B, D) [D = # of Classes]
        student_logits == (B, D) [D = # of Classes]
        teacher_features == list of (B, N, C) [N = (hw) tokens]
        student_features == list of (B, N, C) [N = (hw) tokens] features
        labels == (B, D) [D = # of Classes] either one hot vector or labelled smooth (using cut mix augmentation)
        student_logits_teacher_features == (B, D) [logits generated using Teacher Backbone and Student Classifier Head]
        '''
        base_loss = self.base_criterion(student_logits, labels)
        base_loss = self.alpha * base_loss

        logit_loss = 0   
        aux_loss = 0
        if self.aux_loss is not None :
            for index in self.features_index:
                aux_loss += self.aux_loss(  self.spatial_avg(teacher_features[index]), self.spatial_avg(student_features[index]), labels=labels  )
                aux_loss = self.beta * aux_loss

        if self.logit_loss is not None :
            logit_loss = self.logit_loss(teacher_logits, student_logits, student_logits_teacher_features=student_logits_teacher_features)
            logit_loss = self.gamma * logit_loss

        loss = aux_loss + base_loss + logit_loss
        
        return loss


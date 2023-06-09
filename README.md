# Distillation-Loss
Implementation of Various Distillation Losses (2019-2023) 

## Note 

One Teacher multiple students may fail in some cases. simple loss functions will work perfectly fine though. Need to tweak to the code I'm too lazy to do anything about it right now. I will fix them later.

### Description

`multiple_lrs`: One Teacher and Corresponding multiple Students   
`spatial_avg`: Spatilly average the spatial tokens or not   
`tau`: Temperature Param to soften Teacher logits (making it easy for student to reconstruct teacher logits)  
`features_index`: Number of layers(starting from end) to reconstruct via student  
`mode` : Hyper param for a particular loss being invoked

`lambda1`, `lambda2`, `lambda3`: hyper param weights of loss within auxillary loss   
`alpha` : weight of normal cross entorpy loss of student  
`beta` :  weight of Auxillary loss applied on Student Features 
`gamma` :  weight of Logit loss applied on Student logits when comparing against Teacher Logits 

## Ussage

#### [Improved Knowledge Distillation for Training Fast Low Resolution Face Recognition Model, 2019](https://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Wang_Improved_Knowledge_Distillation_for_Training_Fast_Low_Resolution_Face_Recognition_ICCVW_2019_paper.pdf)

 - L_HARD: Cross Entropy Loss: alpha == lambda1(in the paper)  
 - L_SOFT: Logit loss `soft_entropy`, gamma == lambda2 (in the paper)
    - Only compare students logits with teacher logits `lambda1=1.0`, `lambda2=0`
    - Temperature `tau=2`
    - Soften students logit as well `mode="student_softening"`
 - L_FEATURE : Auxillary Loss, `l2` or `l1`, self.beta == lambda3 (in the paper)
    - features_index=[-1] (Features reconstruction only at last layer)
    - Paper propeses `MK-MMD` loss
    
```
from loss import DistillationLoss
Diss_Loss = DistillationLoss(
    logit_loss="soft_entropy", 
    lambda1=1.0, lambda2=0, tau=2, mode="student_softening",
    aux_loss="l1", features_index=[-1],  
    alpha=1.0 , beta =1.0, gamma =1.0, spatial_avg="mean",
)

Diss_Loss = DistillationLoss(
    logit_loss="soft_entropy", 
    lambda1=1.0, lambda2=0, tau=2, mode="student_softening",
    aux_loss="MK-MMD", features_index=[-1],  
    alpha=1.0 , beta =1.0, gamma =1.0, spatial_avg="mean",
)
```

#### [Low-Resolution Face Recognition in the Wild with Mixed-Domain Distillation, 2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8919361)
   
```
from loss import DistillationLoss
Diss_Loss = DistillationLoss(aux_loss="l1", features_index=[-1], alpha=1.0, beta =1.0, spatial_avg="mean")
```

#### [Fewer-Shots and Lower-Resolutions: Towards Ultrafast Face Recognition in the Wild, 2019](https://dl.acm.org/doi/pdf/10.1145/3343031.3351082?casa_token=wdms_EHiPZEAAAAA:KQtFlBNkOZIq4Ubri935TxatOEOBWPGASmIO1LdoKqpY619lCuia4DUBqAx5k1YMxw_lwk7LNEM6)

 - For every Teacher, there are d(=9) student outputs `multiple_lrs=9`
 - Size of labels  == Size of students 
 - During fine-tunning use relations loss `relational`

```
from loss import DistillationLoss
Diss_Loss = DistillationLoss(aux_loss="l2", features_index=[-1],  alpha=1.0, beta =1.0, spatial_avg="mean", multiple_lrs=9)

Diss_Loss = DistillationLoss(aux_loss="relational", features_index=[-1],  alpha=1.0, beta =1.0, spatial_avg="mean", no_teacher=True)
```





## Testing the above loss 

 - B == 2 
 - Num Classes == 200 
 - Last layer features : 2, 7,7, 2048 --> 2,49,2048 
 - labels is softmaxed() 
 - student_logits_teacher_features. Can be `None` as well 

#### 1 TEACHER 1 STUDENT
```
import torch
teacher_logits = torch.rand(2, 200)
teacher_features = [torch.rand(2, 196, 1024), torch.rand(2, 49, 2048)] 
student_logits_teacher_features = torch.rand(2, 200)
labels = torch.rand(2, 200).softmax(-1)
student_logits = torch.rand(2, 200)
student_features = [torch.rand(2, 196, 1024), torch.rand(2, 49, 2048)] 

Diss_Loss(
    teacher_logits=teacher_logits, teacher_features=teacher_features, student_logits=student_logits, student_features=student_features, labels=labels, student_logits_teacher_features=student_logits_teacher_features,
)
```
#### 1 TEACHER d STUDENT
```
import torch
teacher_logits = torch.rand(2, 200)
teacher_features = [torch.rand(2, 196, 1024), torch.rand(2, 49, 2048)] 
student_logits = torch.rand(2, 9, 200)
student_features = [torch.rand(2, 9, 196, 1024), torch.rand(2, 9, 49, 2048)] 
labels = torch.rand(2, 9, 200).softmax(-1)
Diss_Loss(
    teacher_logits=teacher_logits, teacher_features=teacher_features, student_logits=student_logits, student_features=student_features, labels=labels, student_logits_teacher_features=student_logits_teacher_features,
)
```
#### No TEACHER, just fine-tunning STUDENT

```
import torch
student_logits = torch.rand(2, 200)
labels = torch.rand(2, 200).softmax(-1)
student_features = [torch.rand(2, 196, 1), torch.rand(2, 49, 1)] 
Diss_Loss(
    teacher_logits=None, teacher_features=None, student_logits=student_logits, student_features=student_features, labels=labels, student_logits_teacher_features=None,
)
```
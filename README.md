# Distillation-Loss
Implementation of Various Distillation Losses (2019-2023) 

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
    alpha=1.0 , beta =1.0, gamma =1.0, 
)

Diss_Loss = DistillationLoss(
    logit_loss="soft_entropy", 
    lambda1=1.0, lambda2=0, tau=2, mode="student_softening",
    aux_loss="MK-MMD", features_index=[-1],  
    alpha=1.0 , beta =1.0, gamma =1.0, 
)
```


## Testing the above loss 

```
# B == 2 || Num Classes == 200 || Last layer features : 2, 7,7, 2048 --> 2,49,2048 || labels is softmaxed() || student_logits_teacher_features Can be None as well 

import torch
teacher_logits = torch.rand(2, 200)
teacher_features = [torch.rand(2, 49, 2048), torch.rand(2, 196, 1024)] 
student_logits = torch.rand(2, 200)
student_features = [torch.rand(2, 49, 2048), torch.rand(2, 196, 1024)] 
labels = torch.rand(2, 200).softmax(-1)
student_logits_teacher_features = torch.rand(2, 200)

Diss_Loss(
    teacher_logits=teacher_logits, teacher_features=teacher_features, student_logits=student_logits, student_features=student_features, labels=labels, student_logits_teacher_features=student_logits_teacher_features,
)
```
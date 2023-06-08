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

### Ussage

#### Improved Knowledge Distillation for Training Fast Low Resolution Face Recognition Model, 2019

```
Diss_Loss = DistillationLoss(

)

```


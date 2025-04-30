## pip install -r requirements.txt
## Pruning
 python prune.py
 ## After prunning, nên train lại : 
 python train.py (với path là file vừa được prunning)

 ## Distillation
 python train_distillation.py

 ## finally, quantize ( tạm thời push mỗi post training)
 python quantize.py

 ## Chú ý, nên dùng new_dataloader để giải quyết vấn đề imbalance của COCO
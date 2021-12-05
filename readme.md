# channel prune for ResNet

you can use this repo to prune your model(only support Res style block, such as BottleNeck and BasicBlock)

use this command to train and test on CIFAR10:

```
python CIFAR_train.py
```

Note:
<br>PruneTool can prune the model with residual, but OldPruneTool ignore the residual(for BottleNeck, OldPruneTool will only prune the first two conv)

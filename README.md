# ERA-SESSION9

## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
           Dropout-4           [-1, 16, 32, 32]               0
         ConvBNAct-5           [-1, 16, 32, 32]               0
            Conv2d-6           [-1, 32, 32, 32]           4,608
       BatchNorm2d-7           [-1, 32, 32, 32]              64
              ReLU-8           [-1, 32, 32, 32]               0
           Dropout-9           [-1, 32, 32, 32]               0
        ConvBNAct-10           [-1, 32, 32, 32]               0
           Conv2d-11           [-1, 32, 32, 32]           9,216
      BatchNorm2d-12           [-1, 32, 32, 32]              64
             ReLU-13           [-1, 32, 32, 32]               0
          Dropout-14           [-1, 32, 32, 32]               0
        ConvBNAct-15           [-1, 32, 32, 32]               0
           Conv2d-16           [-1, 32, 15, 15]           9,216
      BatchNorm2d-17           [-1, 32, 15, 15]              64
             ReLU-18           [-1, 32, 15, 15]               0
          Dropout-19           [-1, 32, 15, 15]               0
        ConvBNAct-20           [-1, 32, 15, 15]               0
           Conv2d-21           [-1, 32, 15, 15]           9,216
      BatchNorm2d-22           [-1, 32, 15, 15]              64
             ReLU-23           [-1, 32, 15, 15]               0
          Dropout-24           [-1, 32, 15, 15]               0
        ConvBNAct-25           [-1, 32, 15, 15]               0
           Conv2d-26           [-1, 52, 15, 15]          14,976
      BatchNorm2d-27           [-1, 52, 15, 15]             104
             ReLU-28           [-1, 52, 15, 15]               0
          Dropout-29           [-1, 52, 15, 15]               0
        ConvBNAct-30           [-1, 52, 15, 15]               0
           Conv2d-31             [-1, 64, 7, 7]          29,952
      BatchNorm2d-32             [-1, 64, 7, 7]             128
             ReLU-33             [-1, 64, 7, 7]               0
          Dropout-34             [-1, 64, 7, 7]               0
        ConvBNAct-35             [-1, 64, 7, 7]               0
           Conv2d-36             [-1, 64, 7, 7]          36,864
      BatchNorm2d-37             [-1, 64, 7, 7]             128
             ReLU-38             [-1, 64, 7, 7]               0
          Dropout-39             [-1, 64, 7, 7]               0
        ConvBNAct-40             [-1, 64, 7, 7]               0
           Conv2d-41             [-1, 64, 7, 7]          36,864
      BatchNorm2d-42             [-1, 64, 7, 7]             128
             ReLU-43             [-1, 64, 7, 7]               0
          Dropout-44             [-1, 64, 7, 7]               0
        ConvBNAct-45             [-1, 64, 7, 7]               0
           Conv2d-46             [-1, 64, 5, 5]           4,096
      BatchNorm2d-47             [-1, 64, 5, 5]             128
             ReLU-48             [-1, 64, 5, 5]               0
          Dropout-49             [-1, 64, 5, 5]               0
        ConvBNAct-50             [-1, 64, 5, 5]               0
           Conv2d-51             [-1, 64, 5, 5]          36,864
      BatchNorm2d-52             [-1, 64, 5, 5]             128
             ReLU-53             [-1, 64, 5, 5]               0
          Dropout-54             [-1, 64, 5, 5]               0
        ConvBNAct-55             [-1, 64, 5, 5]               0
           Conv2d-56             [-1, 10, 5, 5]           5,760
        ConvBNAct-57             [-1, 10, 5, 5]               0
        AvgPool2d-58             [-1, 10, 1, 1]               0
================================================================
Total params: 199,096
Trainable params: 199,096
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.61
Params size (MB): 0.76
Estimated Total Size (MB): 5.38
----------------------------------------------------------------
```

## Target to be Achieved 
[.] Write a new network that
has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
[.] total RF must be more than 44
[.] one of the layers must use Depthwise Separable Convolution
[] one of the layers must use Dilated Convolution
use GAP (compulsory):- add FC after GAP to target #of classes (optional)
use albumentation library and apply:
horizontal flip
shiftScaleRotate
coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
make sure you're following code-modularity (else 0 for full assignment) 
upload to Github

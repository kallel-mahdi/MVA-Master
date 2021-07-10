In this work we adapt the NTS-NET implementation provided in https://github.com/yangze0930/NTS-Net in order to make fine grained classification for the species of 20 birds.

Dataset was small containing 1000 images so we had to use various data augmentation techniques.

Birds photos were taken from really far so we used a Fast-R-CNN to crop the pictures.

```
@inproceedings{Yang2018Learning,
author = {Yang, Ze and Luo, Tiange and Wang, Dong and Hu, Zhiqiang and Gao, Jun and Wang, Liwei},
title = {Learning to Navigate for Fine-grained Classification},
booktitle = {ECCV},
year = {2018}
}
```

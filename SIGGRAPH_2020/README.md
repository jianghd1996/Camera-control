# Example-driven Virtual Cinematography by Learning Camera Behaviors 

This repo provides PyTorch implementation of our paper :

*Example-driven Virtual Cinematography by Learning Camera Behaviors*

SIGGRAPH 2020

![Overview](https://github.com/jianghd1996/Camera-control/blob/master/SIGGRAPH_2020/Figure/teaser.png)

The homepage of the project can be found [here](https://jianghd1996.github.io/publication/sig_2020/). The paper can be found [here](https://jianghd1996.github.io/publication/sig_2020/SIG_2020.pdf).



**To do list**

- [ ] Usage

estimation

- [ ] dataset
- [ ] code
- [ ] pretrain weight

gating prediction

- [ ] dataset

- [ ] code
- [ ] pretrain weight
- [x] code and comments
- [x] pretrained model
- [ ] test scene





## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.6



## Dependencies

```bash
pip install -r requirements.txt
```



 ## Cinematic Estimation



**Dataset**



**Training and Pretrained Model **



Pretrained model with 10 degree noise

[[google drive](https://drive.google.com/file/d/1PpAKJk8OYqP1m_oMr4DhfHDZiGrN7MQV/view?usp=sharing)]  





## Gating Prediction Model


**Dataset**



**Training and Pretrained Model **



Pretrained model with number of expert = 9 

[[google drive](https://drive.google.com/file/d/1-ulS9hXV1T0FjlWZAo2uAbYe8V__Lntq/view?usp=sharing)]     [[baiduyun](https://pan.baidu.com/s/1bgyuupD0-CaeEH5AE_I6aQ)] (password: 2vp3)







## Testing

In test time, we extract camera behaviors in a reference real film clips and retarget it to a new 3D animation. The process has 3 steps. 



**Step 1** is the cinematic feature estimation. 

According to our paper, this part has three steps :

1. Use [LCR-Net](http://lear.inrialpes.fr/src/LCR-Net/) to estimate 2D skeletons from videos
2. Pose association, smoothing and filling missing joints
3. Estimate cinematic features with a neural network



**Step 2** is mapping the film clips to a latent camera behaviors space.

Use the cinematic feature from step 1, input them to the pretrained Gating network to get a sequence of camera behaviors vector.



**Step 3** is apply the extracted behaviors to a new 3D animation.

Use the vector in step 2 to control the weights of Prediction network in every step, input the scene content to the prediction network to get the camera pose of next frame.



Please refer to folder Movie_Analysis for detail usage.



## Acknowledgments


This work was supported in part by the National Key R&D Program of China (2018YFB1403900, 2019YFF0302902).



## Citation


Please cite our work if you find it useful:

```
@article{jiang2020example,
  title={Example-driven Virtual Cinematography by Learning Camera Behaviors},
  author={Hongda, Jiang and Bin, Wang and Xi, Wang and Marc, Christie and Baoquan Chen},
  journal={ACM Transactions on Graphics (TOG)},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```


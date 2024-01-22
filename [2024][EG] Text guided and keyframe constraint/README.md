# Cinematographic Camera Diffusion Model

This repo provides PyTorch implementation of our paper :

*Cinematographic Camera Diffusion Model*

[Hongda Jiang](https://jianghd1996.github.io/), [Xi Wang](https://triocrossing.github.io/), [Marc Christie](http://people.irisa.fr/Marc.Christie/), [Libin Liu](http://libliu.info/),[Baoquan Chen](https://baoquanchen.info/)

Eurographics 2024

The homepage and paper will release after published.

## Prerequisites

The environment requirement for this repo is simple.

- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.8
- Pytorch, torchvision, tqdm, matplotlib, numpy, [CLIP](https://github.com/openai/CLIP)

## Dataset

We provide dataset in [link](https://drive.google.com/file/d/1VxmGy9szWShOKzWvIxrmgaNEkeqGPLJU/view?usp=sharing). The dataset is a numpy dict, where the key 'cam' includes the camera trajectories and 'info' includes the text descriptions.

## Pretrained Model

We provide [weights](https://drive.google.com/file/d/136IZeL4PSf9L6FJ4n_jFM6QFLTDjbvr1/view?usp=sharing) with text only training results. Please create an empty folder `weight` and put the weight file into the folder.

Tips:
If you want to use the pretrained weight, please use *zooms in* and *zooms out* when you want to generate sequence with *pushes in* and *pulls out* since in the training we use these two prompts.

## Inference

Simply run ```python main.py gen``` and the generated sequences will be put in folder `gen`.

We provide a Unity Scene for visualize the result [link](https://drive.google.com/file/d/1zAOJ8zN2hYO-dlQJSNl5uR_JtKapjpM8/view?usp=sharing), the version of the project is 2018.2.13f1. You need to set the file path, shooting target (head), shooting character. Here we provide an example of 'pan' motion with prompt 'The camera pans to the character. The camera switches from right front view to right back view. The character is at the middle center of the screen. The camera shoots at close shot.'.

![camera_parameter](image/Unity_script.png)

![camera_run](image/unity.gif)

## Evaluation

We provide the code of classifier `classify.py`, metric `metric.py`, and LSTM based camera motion generator `LSTM.py`. The training and testing dataset are separated with 9:1 ratio randomly.

## Acknowledgement

This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:

[Conditional Diffusion MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST), [MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model).
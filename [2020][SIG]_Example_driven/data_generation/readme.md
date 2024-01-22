We provide implemented data generator in Unity3D which the users could generate any kinds of camera behaviors with different character movements.

Project link : 

[[google drive](https://drive.google.com/file/d/1i6l2gK2VrRoHX_ivf_DngxNcemdSTxca/view?usp=sharing)] [[baiduyun](https://pan.baidu.com/s/1PJRy2EoMEjQf25EFnMqUFw)] (password: pmq4)



Instructions :

1. In Assets/ScenesII, we have 30 scenes with different character movement path, you could also design your own character path with the Path object in each scene.
2. The Randomize Animation.cs script attached to PlayerA could be used to add noise to character joints, which could be used to increase the training robustness.
3. The Camera Control.cs script attached to Main Camera is used to control camera and generate dataset, in each frame, the code will output the screen position and 3D position of each joints.



data_processing.py could extract the proposed character features and camera features from generated scene file, if you want to use other scenes, remember to update the corresponding skeleton name in the function 'extract_txt'.
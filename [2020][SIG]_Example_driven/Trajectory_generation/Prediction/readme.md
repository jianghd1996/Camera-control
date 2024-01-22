The pipeline of using Unity and Python code offline :

1. unity, download the unity file, switch to target scene, use mode 0 to output scene information to > scene.txt
2. python, use predictor.py to load scene.txt and generate camera sequence > camera.txt
3. unity, use mode 1 to load camera.txt and output global camera coordinate to > scene.txt
4. python, use camera.py to load scene.txt and generate smooth camera sequence > smooth_camera.txt
5. unity, use mode 2 to load smooth camera trajectory and visualize
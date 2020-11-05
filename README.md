# RBVMS

<br>
<br>

<img src="https://github.com/ankursikarwar/RBVMS/blob/master/UI.png" alt="Image4" width="900" height="550"/>     

<br>
<br>

## For non-commercial use only
<br>

## Installation 

### The code has been tested on Ubuntu 18.04 with Python 3.6

1. Install virtualenv

```
sudo apt install -y python3-venv
```

2. Clone the Repo

```
git clone https://github.com/ankursikarwar/RBVMS.git
```

3. Navigate to the sub-folder

```
cd RBVMS
```

4. Initialize virtual environment 

```
python3 -m venv my_env
```

5. Activate the virtual environment

```
source my_env/bin/activate
```

6. Upgrade pip 

```
(my_env) pip install --upgrade pip
```

7. Install the dependencies

```
(my_env) pip install -r requirements.txt
```

8. (Optional) Users may need to change the camera device index based on their webcam configuration. The default camera index is 0. Check line 43 in demo.py and line 42 in demo_visualization.py

9. (Optional) Users can also switch 'on' image enhancement to use the application in low-lighting conditions. See line 44, 45 in demo.py and line 43, 44 in demo_visualization.py

10. Starting the Application

Without Visualization:
```
(my_env) python demo.py
```

With Visualization:
```
(my_env) python demo_visualization.py
```



## References

1. Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.

2. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman (2018). VGGFace2: A dataset for recognising face across pose and age. International Conference on Automatic Face and Gesture Recognition, 2018.

3. Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2015.

4. Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke and Alex Alemi (2016). Inception-v4, Inception-ResNet and the Impact
of Residual Connections on Learning. Computer Vision and Pattern Recognition.

5. Guo, Chunle Guo and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin (2020). Zero-reference deep curve estimation for low-light image enhancement. Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR).


## Credits

Credits for image enhancement code go to https://github.com/Li-Chongyi/Zero-DCE 

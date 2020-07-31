# RBVMS-FE

<br>
<br>

<img src="https://github.com/ankursikarwar/RBVMS-FE/blob/master/UI.png" alt="Image4" width="900" height="550"/>     

<br>
<br>

## Installation 

1. Install virtualenv

```
sudo apt install -y python3-venv
```
2. Create a project folder


```
mkdir RBVMS-FE
cd RBVMS-FE
```
3. Initialize virtual environment 

```
python3 -m venv my_env
```
4. Activate the virtual environment

```
source my_env/bin/activate
```

5. Clone the Repo

```
(my_env) git clone https://github.com/ankursikarwar/RBVMS-FE.git
```

6. Navigate to the sub-folder

```
(my_env) cd RBVMS-FE
```

7. Install the dependencies

```
(my_env) pip install -r requirements.txt
```

8. (Optional) User may need to change camera device index based on their webcam configuration. Default camera index is 0. Check line 41 in demo.py


9. Starting the Application


```
(my_env) python demo.py
```
## References

1. Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.

2. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman (2018). VGGFace2: A dataset for recognising face across pose and age. International Conference on Automatic Face and Gesture Recognition, 2018.

3. Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2015.

4. Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke and Alex Alemi (2016). Inception-v4, Inception-ResNet and the Impact
of Residual Connections on Learning. Computer Vision and Pattern Recognition.


## License


This project is licensed under the MIT License.

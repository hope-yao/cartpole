OpenAI gym implementation of cart-pole example for MAE540. A simple Q learning algorithm is used to control the system.
![example](/../master/readme/animation.gif)

## Requirements
If you are on Ubuntu, you can install prerequisits using commands below:
```
$pip install jupyter
$pip install tensorflow
$pip install gym
```
Otherwise, please refer to these pages for detailed installation guidance.
[ipython](https://ipython.org/ipython-doc/2/install/install.html)

[tensorflow](https://www.tensorflow.org/install/) 

[gym](https://gym.openai.com/docs) 

Besides, if you are on Ubuntu 14.04, you might also need to install ffmpeg player for proper video output:
```
$sudo apt-get install libav-tools
```

To run this repo:
```
$python main.py
```

## Physical problem
How to optimize a system controller when the system model is unknown. 
State space is expanded by the location and speed of cart and angle and angular velocity of the pole (denoted by x, x_dot, theta, theta_dot).
A force with magnitude of 10N is applied to the system.
The force can be in either left or right direction.
The system stays in a given region of state space as long as posible.
It starts with a small purtubation on the system state.
And it will be considered as a failure trial whenever the state variables exit the region (-2.4, 2.4, -0.21, 0.21).

More details about the setup of the pysical environment can be found as [gym](https://github.com/openai/gym/wiki/CartPole-v0)
Details on the governing equations for single pole can be found at [this technical report](https://pdfs.semanticscholar.org/3dd6/7d8565480ddb5f3c0b4ea6be7058e77b4172.pdf)
Corresponding equations for multiple poles can be found at [this paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=155416)



## Controller

A two layer MLP is used as controller. The network struction is:

![MLP](/../master/assets/cart-pole-controller.png)

Stochastic gradient descent with momentum is used to train this network:

![SGD](https://wikimedia.org/api/rest_v1/media/math/render/svg/4895d44c0572fb2988f2f335c28cc055a7f75fa0)

Its paramters can be played with.

Rewards is normalized to speed up the training process as described in [this paper](https://arxiv.org/abs/1602.07714)

## Results
By keep default parameters, we can get convergence curve similar to: 

![iteration](/../master/readme/iteration.png)

> The small window shows the location of states 





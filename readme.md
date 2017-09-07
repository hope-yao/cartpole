OpenAI gym implementation of cart-pole example for MAE540. A simple Q learning algorithm is used to control this system.
![example](/../master/readme/animation.gif)

## Requirements
If you are under Ubuntu, you can install prerequisits using commands below:
```
$pip install jupyter
$pip install tensorflow
$pip install gym
```
Otherwise, please refer to these pages for detailed installation guidance.

[ipython](https://ipython.org/ipython-doc/2/install/install.html)

[tensorflow](https://www.tensorflow.org/install/) 

[gym](https://gym.openai.com/docs) 

Besides, if your Ubuntu is 14.04, you might also need to install ffmpeg player for proper video output:
```
$sudo apt-get install libav-tools
```

Simplest way to run this repo:
```
$python main.py
```
You can also specify your own learning rate and momentum for the training process by replacing \<lr\> and \<momentum\> with your own values: 
```
$python main.py <lr> <momentum>
```

## Physical problem
State space is expanded by the location and speed of cart and angle and angular velocity of the pole (denoted by x, x_dot, theta, theta_dot).
It starts with a small purtubation on the system state.
A force in either left or right direction is applied to the cart to balance the pole.
The trial will be terminated if the system has been balancing for more than 4 seconds. And a trial will be terminated as well whenever the state variables exit a given threshold. 
The **objective** of the control algorithm is to make system to stay in a given region of state space as long as posible.

| Left-aligned | Center-aligned |
|  :---:       |     :---:      |    
| mass cart   | 1.0     | 
| mass pole     | 0.1       |
| pole length     | 0.5       |
| force      | 10.0       |
| delta_t      | 0.02       |
| theta_threshold      | 12 (degrees)       |
| delta_t      | 2.4       |

It is also fine if you wish to change these paramters to your own values. You can find them under gym_installation_dir/envs/classic_control/cartpole.py.

More details about the setup of the pysical environment can be found in the [gym documents](https://github.com/openai/gym/wiki/CartPole-v0).
Details on how to derive the governing equations for single pole can be found at [this technical report](https://pdfs.semanticscholar.org/3dd6/7d8565480ddb5f3c0b4ea6be7058e77b4172.pdf).
Corresponding equations for how to generalize this to multiple poles can also be found at [this paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=155416)



## Controller

A two layer MLP is used as controller. The network struction is:

![MLP](/../master/readme/cart-pole-controller.png)

It is found that a single layer is already sufficient in this environment setting. If needed, you can replace the network with a more complex one. Stochastic gradient descent with momentum is used to train this network:

![SGD](https://wikimedia.org/api/rest_v1/media/math/render/svg/4895d44c0572fb2988f2f335c28cc055a7f75fa0)

You can play with its paramters by using your own inputs to main.py.


## Results
By keep default parameters, we can get a convergence curve similar to: 

![iteration](/../master/readme/iteration.png)

> X axis of the big window is the total reward a policy achieves, Y axis of it is the number of current training epoch. The small window shows the relative value of cart position and pole angle w.r.t. its maximum accepted value. It can be seen that the training is occilating but achieved good policy in the end.

## DQN
Rewards is normalized to speed up the training process as described in [this paper](https://arxiv.org/abs/1602.07714)

## TODO
- Move all code to ipynb
- Add more intro to RL


<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

## Gradient Descent Application: Pole Balancing

### Introduction
![example](/../master/readme/animation.gif)
In this application, you will learn how to use OpenAI gym to create a 
controller for the classic pole balancing problem. The problem will be solved
using Reinforcement Learning. While this topic requires much involved 
discussion, here we present a simple formulation of the problem that can be
efficiently solved using gradient descent. Also note that pole balancing
can be solved by classic control theory, e.g., through a PID controller.
However, our solution does not require linearization of the system.

### Get Ready
#### Downloading this repo
To download the [code][polecode], you can either click on the green button to download 
as a zip, or use [git][giturl]. Please see a git tutorial [here][gitdoc].

#### Install prerequisites

1. You will first need to install [Python 3.5][pythonurl]. Check if python is 
correctly installed by type in the command line ```python```.
2. Install TensorFlow (CPU version) [here](https://www.tensorflow.org/install/).
3. Once done, go to the folder where you hold this code, type in
```
pip install -r requirement.txt
``` 
This should install all dependancies.
4. You can then run the code by typing in 
```
python main.py
```

<!--For **Linux** users, you can install prerequisits using the commands below:-->
<!--```-->
<!--$pip install jupyter-->
<!--$pip install tensorflow-->
<!--$pip install gym-->
<!--```-->

### The design problem
#### Problem statement
The simple pole balancing (inverse pendulum) setting consists of a pole 
and a cart. The system states are the cart displacement $$x$$, 
cart velocity $$\dot{x}$$, pole angle $$\theta$$, and pole angular velocity 
$$\dot{\theta}$$. The parameters of the default system is as follows:

| Left-aligned | Center-aligned |
|  :---:       |     :---:      |    
| mass cart   | 1.0     | 
| mass pole     | 0.1       |
| pole length     | 0.5       |
| force      | 10.0       |
| delta_t      | 0.02       |
| theta_threshold      | 12 (degrees)       |
| delta_t      | 2.4       |

The system equations are ***.

The controller takes in the system states, and outputs a fixed force on the cart
to either left or right. The controller needs to be designed so that within
4 seconds, the pole angle does not exceed 12 degrees, and the cart displacement
does not exceed 2.4 unit length. A trial will be terminated if the system 
is balanced for more than 4 seconds, or any of the constraints are violated. 
<!--The **objective** of the control algorithm is to make system to stay in a given region of state space as long as posible. Default physical parameters of this problem are:-->

#### The optimization problem
Here we learn a controller in a model-free fashion, i.e., the controller 
is learned without understanding of the dynamical system. We first introduce
the concept of a **Markov Decision Process**: An MDP contains a *state space*,
an *action space*, a *transition function*, a *reward function*, and 
a decay parameter $$\gamma$$. In our case,
the state space contains all possible combinations of $$x$$, $$\dot{x}$$, 
$$\theta$$, $$\dot{\theta}$$. The action space contains the force to the left, 
and the force to the right. The transition function $$s_{k+1} = T(s_k,a_k)$$
computes the next state $$s_{k+1}$$ based on the current state $$s_k$$ and 
the action $$a_k$$. In our case, the transition is given by the system equations.
The reward function defines an instantaneous reward $$r_k = r(s_k)$$.
In our case, we define reward as 1 when the system does not fail, or 0 
otherwise. The decay parameter $$\gamma$$ defines a long term 
*value* of the controller $$\pi$$: $$V_k(\pi) = r_k + 
\gamma T(s_k,a_k)V_{k+1}(\pi)$$. $$\gamma$$ describes how important 
future rewards are to the current control decision: larger decay
leads to more greedy decisions.
  
The goal of optimal control is thus to find a controller $$\pi$$ that maximizes 
the expectation $$\mathbb{E}[V_0(\pi)]$$. Specifically, we define the 
controller as a function of the states that outputs a number between 0 and 1: $$\pi(s,w)$$. 
This number is treated as a probability for choosing action 0 (say, force to the left), 
and thus the probability for the other action is $$1-\pi(s,w)$$.
Thus $$V_0(\pi)$$ becomes a random variable parameterized by $$w$$.

#### Q learning
When the transition is not known to the controller, one can use 
**Q learning** to indirectly solve the optimization problem. I will skip 
details and go directly to the solution. Given a trail with $$K$$ time steps
based on the current controller, we collect the instantaneous 
rewards $$r_k$$, actions $$a_k$$, and the controller outputs $$\pi_k$$. 
We minimize the following loss function

$$ f(w) = -\sum_{k=1}^K (\sum_{j=k}^K \gamma^{j-k}r_j) (a_k\log(\pi_k)+(1-a_k)\log(1-\pi_k))$$.

Essentially, this objective is maximized when control decisions all lead
to high value. Thus by tuning $$w$$, we correct the mistakes we made in the 
trial (high value from unlikely move, or low value from favored move).

In the code, you may notice that the discounted rewards 
$$\sum_{j=k}^K \gamma^{j-k}r_j$$ are normalized.
According to [this paper](https://arxiv.org/abs/1602.07714), this 
speeds up the training process.

#### Controller model
The controller is modeled as a single-layer neural network:

![MLP](/../master/readme/cart-pole-controller.png)

It is found that a single layer is already sufficient for this environment setting. 
If needed, you can replace the network with more complicated ones. 
<!--Stochastic gradient descent with momentum is used to train this network:-->
<!--![SGD](https://wikimedia.org/api/rest_v1/media/math/render/svg/4895d44c0572fb2988f2f335c28cc055a7f75fa0)-->
<!--You can play with its paramters by using your own inputs to main.py.-->

#### Training algorithm
Due to the probabilistic nature of the value function, we minimize an averaged
loss $$F(w) = \sum_{t=1}^T f_t(w)$$ over $$T$$ trials. This is done by simply
running the simulation $$T$$ times, recording all data, and calculate the gradient
of $$F(w)$$. Notice that the gradient in this case will be stochastic, in the sense that
we only use $$T$$ random samples to approximate it, rather than finding the theoretical
mean of $$\nabla_w F(w)$$ (which does not have an analytical form anyway).
The implementation of the gradient descent is [ADAM][adam], 
which we will discuss later in the class. 

### Results
With default problem settings, we can get a convergence curve similar this: 

![iteration](/../master/readme/iteration.png)

> Y axis of the main plot is the total reward a policy achieves, 
X axis is the number of training epochs. 
The small window shows the normalized trajectory of cart positions and 
pole angles in the most recent trial. It can be seen that the learning
achieves a good controller in the end.

To store videos, you will need to uncomment the line:
```
# self.env = wrappers.Monitor(self.env, dir, force=True, video_callable=self.video_callable)
```
By doing this, a serial of the simulation videos will be saved in the folder ```/tmp/trial```.

### Generalization of the problem
You can change problem parameters in ```gym_installation_dir/envs/classic_control/cartpole.py```.
More details about the setup of this physical environment can be found 
in the [gym documents](https://github.com/openai/gym/wiki/CartPole-v0).
Details on how to derive the governing equations for single pole can be 
found at [this technical report](https://pdfs.semanticscholar.org/3dd6/7d8565480ddb5f3c0b4ea6be7058e77b4172.pdf).
Corresponding equations for how to generalize this to multiple poles 
can also be found at [this paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=155416)




<!--Besides, if your Ubuntu is 14.04, you might also need to install ffmpeg player for proper video output:-->
<!--```-->
<!--$sudo apt-get install libav-tools-->
<!--```-->

<!--The simplest way to run this repo is:-->
<!--```-->
<!--$python main.py-->
<!--```-->
<!--You can also specify your own learning rate and momentum for the training process by replacing \<lr\> and \<momentum\> with your own values: -->
<!--```-->
<!--$python main.py <lr> <momentum>-->
<!--```-->

## TODO
- Move all code to ipynb
- Add more intro to RL

[ipython](https://ipython.org/ipython-doc/2/install/install.html)

[tensorflow](https://www.tensorflow.org/install/) 

[gym](https://gym.openai.com/docs) 
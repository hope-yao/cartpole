
import sys
from pylab import *
# import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import gym
import numpy as np
import tensorflow.contrib.layers as layers
from gym import wrappers


def discounted_reward(rewards, gamma):
    """Compute the discounted reward."""
    ans = np.zeros_like(rewards)
    running_sum = 0
    # compute the result backward
    for i in reversed(range(len(rewards))):
        running_sum = running_sum * gamma + rewards[i]
        ans[i] = running_sum
    return ans


class Agent(object):
    def __init__(self, input_size=4, hidden_size=2, gamma=0.95,
                 action_size=2, alpha=0.1, dir='tmp/trial/'):
        self.env = gym.make('CartPole-v0')
        # self.env = wrappers.Monitor(self.env, dir, force=True, video_callable=self.video_callable)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.action_size = action_size
        self.alpha = alpha
        # save the hyper parameters
        self.params = self.__dict__.copy()
        # placeholders
        self.input_pl = tf.placeholder(tf.float32, [None, input_size])
        self.action_pl = tf.placeholder(tf.int32, [None])
        self.reward_pl = tf.placeholder(tf.float32, [None])
        # a two-layer fully connected network
#        hidden_layer = layers.fully_connected(self.input_pl,
#                                              hidden_size,
#                                              biases_initializer=None,
#                                              activation_fn=tf.nn.relu)
        # hidden_layer = layers.fully_connected(hidden_layer,
        #                                       hidden_size,
        #                                       biases_initializer=None,
        #                                       activation_fn=tf.nn.relu)
        self.output = layers.fully_connected(self.input_pl,
                                             action_size,
                                             biases_initializer=None,
                                             activation_fn=tf.nn.softmax)
        # responsible output
        self.one_hot = tf.one_hot(self.action_pl, action_size)
        self.responsible_output = tf.reduce_sum(self.output * self.one_hot, axis=1)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_output) * self.reward_pl)
        # training variables
        variables = tf.trainable_variables()
        self.variable_pls = []
        for i, var in enumerate(variables):
            self.variable_pls.append(tf.placeholder(tf.float32))
        self.gradients = tf.gradients(self.loss, variables)
        solver = tf.train.AdamOptimizer(learning_rate=alpha)
        solver = tf.train.MomentumOptimizer(learning_rate=alpha,momentum=0.9)
        self.update = solver.apply_gradients(zip(self.variable_pls, variables))


    def video_callable(self, episode_id):
        return episode_id % 50 == 0

    def next_action(self, sess, feed_dict, greedy=False):
        """Pick an action based on the current state.
        Args:
        - sess: a tensorflow session
        - feed_dict: parameter for sess.run()
        - greedy: boolean, whether to take action greedily
        Return:
            Integer, action to be taken.
        """
        ans = sess.run(self.output, feed_dict=feed_dict)[0]
        if greedy:
            return ans.argmax()
        else:
            return np.random.choice(range(self.action_size), p=ans)

    def show_parameters(self):
        """Helper function to show the hyper parameters."""
        for key, value in self.params.items():
            print(key, '=', value)




def one_trial(agent, sess, grad_buffer, reward_itr, i, render = False):
    for idx in range(len(grad_buffer)):
        grad_buffer[idx] *= 0
    s = agent.env.reset()
    state_history = []
    reward_history = []
    action_history = []
    current_reward = 0

    while True:
        feed_dict = {agent.input_pl: [s]}
        greedy = False
        action = agent.next_action(sess, feed_dict, greedy=greedy)
        snext, r, done, _ = agent.env.step(action)
        if render and i % 500 == 0:
            agent.env.render()
        current_reward += r
        state_history.append(s)
        reward_history.append(r)
        action_history.append(action)
        s = snext
        if done:
            reward_itr += [current_reward]
            rewards = discounted_reward(reward_history, agent.gamma)
            # normalizing the reward really helps
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            feed_dict = {
                agent.reward_pl: rewards,
                agent.action_pl: action_history,
                agent.input_pl: np.array(state_history)
            }
            episode_gradients = sess.run(agent.gradients,feed_dict=feed_dict)
            for idx, grad in enumerate(episode_gradients):
                grad_buffer[idx] += grad

            feed_dict = dict(zip(agent.variable_pls, grad_buffer))
            sess.run(agent.update, feed_dict=feed_dict)
            # reset the buffer to zero
            for idx in range(len(grad_buffer)):
                grad_buffer[idx] *= 0
            break
    return state_history


def several_trials(agent, sess, grad_buffer, reward_itr, i, render = False):
    update_every = 3
    for j in range(update_every):
        s = agent.env.reset()
        state_history = []
        reward_history = []
        action_history = []
        current_reward = 0
        while True:
            feed_dict = {agent.input_pl: [s]}
            greedy = False
            action = agent.next_action(sess, feed_dict, greedy=greedy)
            snext, r, done, _ = agent.env.step(action)
            if render and i % 50 == 0:
                agent.env.render()
            current_reward += r
            state_history.append(s)
            reward_history.append(r)
            action_history.append(action)
            s = snext
            if done:
                reward_itr += [current_reward]
                rewards = discounted_reward(reward_history, agent.gamma)
                # normalizing the reward really helps
                rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                feed_dict = {
                    agent.reward_pl: rewards,
                    agent.action_pl: action_history,
                    agent.input_pl:  np.array(state_history)
                }
                episode_gradients = sess.run(agent.gradients,
                                             feed_dict=feed_dict)
                for idx, grad in enumerate(episode_gradients):
                    grad_buffer[idx] += grad

                if i % update_every == 0:
                    feed_dict = dict(zip(agent.variable_pls, grad_buffer))
                    sess.run(agent.update, feed_dict=feed_dict)
                    # reset the buffer to zero
                    for idx in range(len(grad_buffer)):
                        grad_buffer[idx] *= 0
                break

    return state_history

def animate_itr(i,*args):
    agent, sess, grad_buffer, reward_itr, sess, grad_buffer, agent, obt_itr, render = args

    state_history = several_trials(agent, sess, grad_buffer, reward_itr, i, render)
    xlist = [range(len(reward_itr))]
    ylist = [reward_itr]
    for lnum, line in enumerate(lines_itr):
        line.set_data(xlist[lnum], ylist[lnum])  # set data for each line separately.

    if len(reward_itr) % obt_itr == 0:
        x_mag = 2.4
        y_mag = 30 * 2 * math.pi / 360
        # normalize to (-1,1)
        xlist = [np.asarray(state_history)[:,0] / x_mag]
        ylist = [np.asarray(state_history)[:,2] / y_mag]
        lines_obt.set_data(xlist, ylist)
        tau = 0.02
        time_text_obt.set_text('physical time = %6.2fs' % (len(xlist[0])*tau))

    return (lines_itr,) + (lines_obt,) + (time_text_obt,)


def get_fig(max_epoch):
    fig = plt.figure()
    ax_itr = axes([0.1, 0.1, 0.8, 0.8])
    ax_obt = axes([0.5, 0.2, .3, .3])

    # able to display multiple lines if needed
    global lines_obt, lines_itr, time_text_obt
    lines_itr = []
    lobj = ax_itr.plot([], [], lw=1, color="blue")[0]
    lines_itr.append(lobj)
    lines_obt = []

    ax_itr.set_xlim([0, max_epoch])
    ax_itr.set_ylim([0, 220])#([0, max_reward])
    ax_itr.grid(False)
    ax_itr.set_xlabel('trainig epoch')
    ax_itr.set_ylabel('reward')

    time_text_obt = []
    ax_obt.set_xlim([-1, 1])
    ax_obt.set_ylim([-1, 1])
    ax_obt.set_xlabel('cart position')
    ax_obt.set_ylabel('pole angle')
    lines_obt = ax_obt.plot([], [], lw=1, color="red")[0]
    time_text_obt = ax_obt.text(0.05, 0.9, '', fontsize=13, transform=ax_obt.transAxes)
    return fig, ax_itr, ax_obt, time_text_obt


def main():
    obt_itr = 10
    max_epoch = 3000
    render = True
    dir = 'tmp/trial/'

    if len(sys.argv)==1:
        lr, momentum = 0.2, 0.95
        print("Using default learning rate = 0.2 and momentum = 0.95.")
    else:
        try:
            lr,momentum = sys.argv[1:]
            print("Using customized learning rate = %f and momentum = %f." %(lr,momentum))
        except :
            print("Oops! Program requires two variables.  Try again...")
            exit()
    fig, ax_itr, ax_obt, time_text_obt = get_fig(max_epoch)
    global reward_itr
    reward_itr = []
    agent = Agent(hidden_size=24, alpha=lr, gamma=momentum, dir=dir)
    agent.show_parameters()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)
    tf.global_variables_initializer().run(session=sess)
    grad_buffer = sess.run(tf.trainable_variables())
    tf.reset_default_graph()

    args = [agent, sess, grad_buffer, reward_itr, sess, grad_buffer, agent, obt_itr, render]
    ani = animation.FuncAnimation(fig, animate_itr,fargs=args)
    plt.show()

if __name__ == "__main__":
   main()

# Set up formatting for the movie files
# print('saving animation...')
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)

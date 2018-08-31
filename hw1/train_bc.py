"""
The prerequisites for running this:
The following folders should already exist in the current path, even empty:
    experts -> Contains all the expert policies, should not be empty
    expert_data -> Contains all the reference (o,a) pair set obtain from running expert policies, if empty, run_expert.py
    ckpt -> Contains checkpoints during training, fills up if training is done
    bc_agent -> Contains stored models after training for each individual task using simple behavioral cloning
    dagger_agent -> Contains stored models after training for each individual task using DaAger
    log -> Contains all the necessary outputs for generating the plots
examples:
    Training a model for a specific task (The stored model is going to be located in either ./bc_agent or ./dagger_agent depending
    on which algorithm used during training). For more details on flags read main()
    >>> python train_bc.py expert_data/Hopper-v2.pkl Hopper-v2
    >>> python train_bc.py expert_data/Hopper-v2.pkl Hopper-v2 --use_dagger --num_rollouts 10 --num_aggr 5
    Loading the model learned on a specific task (located in either ./bc_agent or ./dagger_agent depending
    on which algorithm used during training). For more details on flags read main()
    >>> python train_bc.py expert_data/Hopper-v2.pkl Hopper-v2 --load_model --model_dir bc_agent
    >>> python train_bc.py expert_data/Hopper-v2.pkl Hopper-v2 --load_model --model_dir dagger_agent

"""
import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import pprint
import random
import math

# utility functions and classes
class BatchGenerator(object):
    def __init__(self, data_set, labels, batch_size):
        self._data_set = data_set
        self._labels = labels
        self._data_size = data_set.shape[0]
        self._batch_size = batch_size
        self._segment = self._data_size // batch_size
        self.last_index = 0

    def next(self):

        if ((self.last_index+1)*self._batch_size > self._data_size):
            data1 = self._data_set[self.last_index * self._batch_size:,:]
            data2 = self._data_set[:((self.last_index+1)*self._batch_size)%self._data_size, :]
            labels1 = self._labels[self.last_index * self._batch_size:, :]
            labels2 = self._labels[:((self.last_index+1)*self._batch_size)%self._data_size, :]
            batch_data = np.concatenate((data1, data2), axis=0)
            batch_labels = np.concatenate((labels1, labels2), axis=0)
        else:
            batch_data = self._data_set[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size, :]
            batch_labels = self._labels[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size, :]

        self.last_index = (self.last_index+1) % (self._segment+1)
        return batch_data, batch_labels

def weight_init(in_size):
    # weight initialization with var = 2 / n
    # http://cs231n.github.io/neural-networks-2/
    return tf.truncated_normal_initializer(0, math.sqrt(1/in_size))

def activation_size(tsr):
    s = tsr.shape
    if (len(s) != 2):
        raise ValueError("Expecting a 2D activation vector, BxN.")
    else:
        return int(s[1])

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def exe(self, data):
        norm_data = data - self.mean
        norm_data /= (self.std + 1e-6)
        return norm_data

def query_expert_policy(expert_policy_file, obs_list):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    observations = []
    actions = []
    for obs in obs_list:
        action = policy_fn(obs[None,:])
        actions.append(action)

    return np.array(actions)


valid_frac = 0.1
display_step = 1
batch_size = 32
# num_epochs is going to get populated later for changing that hyper parameter
num_epochs = 80


# graph definition

learning_rate = 0.005
decay_rate = 0.5
decay_steps = 1000

def constuct_model(graph, dim_list):

    with graph.as_default():
        num_features = dim_list[0]
        num_outputs = dim_list[-1]
        tnsr_in = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name='in')
        tnsr_ref = tf.placeholder(dtype=tf.float32, shape=[None, num_outputs], name='out')

        # with tf.variable_scope('normalizer'):
        #     mu = tf.Variable(tf.zeros([num_features], dtype=tf.float32), name='training_set_mu', trainable=False)
        #     std = tf.Variable(tf.zeros([num_features], dtype=tf.float32), name='training_set_std', trainable=False)
        #     tnsr_in_norm = (tnsr_in - mu) / (std+1e-6)



        global_step = tf.Variable(0, name='global_step', trainable=False)

        starter_learning_rate = learning_rate
        lr = tf.train.exponential_decay(starter_learning_rate, global_step,
                                        decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
        tf.summary.scalar('lr', lr)

        def nn_model(input_data, name='nn_model', reuse=False):
            with tf.variable_scope(name):

                layer = input_data
                for i in range(1, len(dim_list)-1):
                    layer = tf.contrib.layers.fully_connected(layer, dim_list[i],
                                                              reuse=reuse, scope='fc'+str(i),
                                                              activation_fn=tf.nn.relu,
                                                              weights_initializer=weight_init(activation_size(layer)))

                logits = tf.contrib.layers.fully_connected(layer, num_outputs,
                                                           activation_fn=None,
                                                           reuse=reuse, scope='fc_out',
                                                           weights_initializer=weight_init(dim_list[-2]))
            return logits

        tnsr_out = nn_model(tnsr_in, name='train_model', reuse=False)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(0.5*(tf.square(tnsr_out-tnsr_ref)))
            # loss = tf.losses.mean_squared_error(labels=tnsr_ref, predictions=tnsr_out)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        tf.summary.scalar('loss', loss)

        # summarize weights and biases
        all_vars=tf.global_variables()
        def get_var(name):
            for i in range(len(all_vars)):
                if all_vars[i].name.startswith(name):
                    return all_vars[i]
            return None

        tf.summary.histogram('fc1/w', get_var('train_model/fc1/w'))
        tf.summary.histogram('fc2/w', get_var('train_model/fc2/w'))
        tf.summary.histogram('fc_out/w', get_var('train_model/fc_out/w'))
        tf.summary.histogram('fc1/b', get_var('train_model/fc1/b'))
        tf.summary.histogram('fc2/b', get_var('train_model/fc2/b'))
        tf.summary.histogram('fc_out/b', get_var('train_model/fc_out/b'))


        valid_in = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name='valid_in')
        valid_out = tf.placeholder(dtype=tf.float32, shape=[None, num_outputs], name='valid_out')
        # with tf.variable_scope('normalizer', reuse=True):
        #     valid_in_norm = (valid_in - mu) / (std+1e-6)
        valid_logits = nn_model(valid_in, name='train_model', reuse=True)
        with tf.variable_scope('valid_loss'):
            # valid_loss = tf.reduce_mean(tf.reduce_sum(tf.square(valid_logits-valid_out), axis=1), axis=0)
            valid_loss = tf.losses.mean_squared_error(labels=valid_out, predictions=valid_logits)
        tf.summary.scalar("valid_loss", valid_loss)

        merged_summary = tf.summary.merge_all()

    return graph, tnsr_in, tnsr_out, tnsr_ref, loss, optimizer, merged_summary, valid_in, valid_out, valid_loss#, mu, std


# train function
def train(sess, nn,  obs, acts, writer, num_epochs=5000, batch_size=256):
    (graph, tnsr_in, tnsr_out, tnsr_ref, loss, optimizer, merged_summary, valid_in, valid_out, valid_loss) = nn
    # splitting into training and validation set
    indices = np.arange(len(obs))
    bnd = obs.shape[0] - int(obs.shape[0]*valid_frac)
    np.random.shuffle(indices)
    train_ds = obs[indices[:bnd]]
    train_lb = acts[indices[:bnd]]
    valid_ds = obs[indices[bnd:]]
    valid_lb = acts[indices[bnd:]]

    # normalization
    train_std = np.std(train_ds, axis=0)
    train_mean = np.mean(train_ds, axis=0)
    # normalizer = Normalizer(train_mean, train_std)
    # normalizer.exe(train_ds)
    # normalizer.exe(valid_ds)


    batch_gen = BatchGenerator(train_ds, train_lb, batch_size)

    sess.run(tf.global_variables_initializer())
    #mu.assign(train_mean).op.run()
    #std.assign(train_std).op.run()

    for epoch in range(num_epochs):
        avg_loss = 0.
        total_batch = int(train_ds.shape[0]/batch_size)
        # out_logits,l  = sess.run([out_logits, loss], feed_dict={train_in: train_ds, train_out: train_lb})
        # loss = np.mean(np.sum((out_logits-train_lb)**2, axis=1), axis=0)
        # print(loss)
        # print(l)

        # break
        # Loop over all batches
        feed_dict={}
        for i in range(total_batch):

            batch_ins, batch_outs = batch_gen.next()
            feed_dict = {tnsr_in:  batch_ins,
                         tnsr_ref: batch_outs,}
            _, l = sess.run([optimizer, loss], feed_dict=feed_dict)

            # Compute average loss
            avg_loss += l / total_batch
            feed_dict.update({valid_in:  valid_ds,
                              valid_out: valid_lb})

        valid_l, s = sess.run([valid_loss, merged_summary], feed_dict=feed_dict)
        writer.add_summary(s, epoch)
        # Display logs per epoch step
        if (epoch) % display_step == 0:
            print("Epoch:", '%04d' % (epoch), "loss=", "{:.9f}".format(avg_loss))
            print("validation loss at this step: {}".format(valid_l))
            store_nn(sess, 'ckpt/checkpoint.ckpt')

def store_nn(sess, fname):
    all_vars = tf.global_variables()
    saver = tf.train.Saver(all_vars)
    # if not os.path.exists('bc_agent'):
    #     os.mkdir('bc_agent')
    # file = os.path.join('bc_agent', fname + '.ckpt')
    saver.save(sess, fname)

def load_nn(sess, fname):
    # file = os.path.join('bc_agent', fname + '.ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, fname)

def run_bc(sess, nn, args, log_file=None):
    (graph, tnsr_in, tnsr_out, tnsr_ref, loss, optimizer, merged_summary, valid_in, valid_out, valid_loss) = nn
    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            feed = {tnsr_in: obs[None, :]}
            action = sess.run(tnsr_out, feed_dict=feed)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    if log_file != None:
        data = dict(
            hyperparams=list(),
            returns_list=list(),
        )

        if os.path.exists(log_file):
            with open(log_file, 'rb') as rf:
                data = pickle.load(rf)

        data['hyperparams'].append(num_epochs)
        data['returns_list'].append(returns)

        with open(log_file, 'wb') as wf:
            pickle.dump(data, wf)

    return np.array(observations)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--resume_training', action='store_true')

    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument("--use_dagger", action='store_true')
    parser.add_argument("--num_aggr", type=int, default=5,
                        help='Number of aggregations')
    parser.add_argument("--model_dir", type=str, default='bc_agent')
    parser.add_argument("--log_file", type=str)
    args = parser.parse_args()

    import gym
    env = gym.make(args.envname)

    with open(args.expert_data_file, 'rb') as f:
        data = pickle.loads(f.read())

    expert_obs = data['observations']
    expert_acts = data['actions']
    expert_acts = np.array([sample.flatten()for sample in expert_acts])

    pprint.pprint(expert_obs[0].shape)
    pprint.pprint(expert_acts[0].shape)

    num_features=env.observation_space.shape[0]
    num_outputs=env.action_space.shape[0]

    dim_list = [num_features, 20, 20, 20, num_outputs]

    g = tf.Graph()
    nn = constuct_model(g, dim_list)
    with tf.Session(graph=g) as sess:
        writer = tf.summary.FileWriter('bc_nn', sess.graph)
        writer.add_graph(graph=tf.get_default_graph())

        if args.load_model:
            # no need for training, just load the model
            load_nn(sess, os.path.join(args.model_dir, args.envname + '.ckpt'))
        elif args.use_dagger:
            # use dagger for training
            obs_pool = expert_obs
            act_pool = expert_acts
            for i in range(args.num_aggr):
                print("[DAGGER] round:%i" %(i+1))
                print(20*"-"+"/ training/ " + 20*"-")
                train(sess, nn, obs_pool, act_pool, writer, num_epochs=20, batch_size=256)
                observations = run_bc(sess, nn, args)
                print("[DAGGER] adding {} samples".format(len(observations)))
                actions = query_expert_policy(os.path.join('experts', args.envname + '.pkl'), observations)
                actions = np.array([sample.flatten() for sample in actions])

                obs_pool = np.concatenate((obs_pool, observations), axis=0)
                act_pool = np.concatenate((act_pool, actions), axis=0)

                store_nn(sess, os.path.join('dagger_agent', args.envname + '.ckpt'))
        else:
            # use simple behavioural cloning
            train(sess, nn, expert_obs, expert_acts, writer, num_epochs=num_epochs, batch_size=256)
            store_nn(sess, os.path.join('bc_agent', args.envname + '.ckpt'))

        run_bc(sess, nn, args, log_file=args.log_file)


if __name__ == '__main__':
    # for num_epochs in np.arange(5, 55,5):
    #     print('num_epoch = %d' %num_epochs)
    main()

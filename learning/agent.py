"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import os
import numpy as np
import tensorflow as tf
import a3c
import env


S_INFO = len(env.PERF_LABELS) + 1 # previous_action, measurements
S_LEN = env.ACTIONS_NUM_SAMPLES # take how many frames in the past
A_DIM = len(env.ACTIONS)
ACTIONS = env.ACTIONS
DEFAULT_ACTION = 0

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 10 # take as a train batch
MODEL_SAVE_INTERVAL = 10
BUFFER_NORM_FACTOR = 10.0

RANDOM_SEED = 42
RAND_RANGE = 1000000
GRADIENT_BATCH_SIZE = 16
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# log in format of timestamp, config, throughput, lat_50, lat_80, lat_99,
# reward, entropy
NN_MODEL = None


def main():

    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    simulator = env.Simulator()

    with tf.Session() as sess, open(LOG_FILE, 'wb') as log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0

        s_batch = []
        a_batch = []
        r_batch = []
        entropy_record = []

        actor_gradient_batch = []
        critic_gradient_batch = []

        while True:  # serve stream processing forever
            #########
            # State #
            #########
            # the action is from past measurements
            history = simulator.get_training_measurements()

            state = np.zeros((S_INFO, S_LEN))
            # this should be S_INFO number of terms
            for i, (action, perf) in enumerate(history):
                throughput, latency_50, latency_80, latency_99 = perf
                state[0, -i-1] = action
                state[1, -i-1] = throughput
                state[2, -i-1] = latency_50
                state[3, -i-1] = latency_80
                state[4, -i-1] = latency_99

            ##########
            # Action #
            ##########
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            ##########
            # Reward #
            ##########
            throughput, latency_50, latency_80, latency_99 = simulator.get_performance(action)

            # reward is throughput minus sum of latencies
            reward = (throughput / 100
                     - latency_50
                     - latency_80
                     - latency_99
                     )

            ##############
            # Recordings #
            ##############
            r_batch.append(reward)
            s_batch.append(state)

            action_vec = np.zeros(A_DIM)
            action_vec[action] = 1
            a_batch.append(action_vec)

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(epoch) + '\t'
                           + str(action) + '\t'
                           + '%.2f' % reward + '\t'
                           + str(throughput) + '\t'
                           + '%.2f' % latency_50 + '\t'
                           + '%.2f' % latency_80 + '\t'
                           + '%.2f' % latency_99 + '\t'
                           + str(entropy_record[-1]) + '\n')
            log_file.flush()

            if len(r_batch) >= TRAIN_SEQ_LEN :  # do training once

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(s_batch=np.stack(s_batch, axis=0),
                                          a_batch=np.vstack(a_batch),
                                          r_batch=np.vstack(r_batch),
                                          terminal=False, actor=actor, critic=critic)
                td_loss = np.mean(td_batch)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                print "===="
                print "Epoch", epoch
                print "TD_loss", td_loss, "Avg_reward", np.mean(r_batch), "Avg_entropy", np.mean(entropy_record)
                print "===="

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: td_loss,
                    summary_vars[1]: np.mean(r_batch),
                    summary_vars[2]: np.mean(entropy_record)
                })

                writer.add_summary(summary_str, epoch)
                writer.flush()

                entropy_record = []

                if len(actor_gradient_batch) >= GRADIENT_BATCH_SIZE:

                    assert len(actor_gradient_batch) == len(critic_gradient_batch)
                    # assembled_actor_gradient = actor_gradient_batch[0]
                    # assembled_critic_gradient = critic_gradient_batch[0]
                    # assert len(actor_gradient_batch) == len(critic_gradient_batch)
                    # for i in xrange(len(actor_gradient_batch) - 1):
                    #     for j in xrange(len(actor_gradient)):
                    #         assembled_actor_gradient[j] += actor_gradient_batch[i][j]
                    #         assembled_critic_gradient[j] += critic_gradient_batch[i][j]
                    # actor.apply_gradients(assembled_actor_gradient)
                    # critic.apply_gradients(assembled_critic_gradient)

                    for i in xrange(len(actor_gradient_batch)):
                        actor.apply_gradients(actor_gradient_batch[i])
                        critic.apply_gradients(critic_gradient_batch[i])

                    actor_gradient_batch = []
                    critic_gradient_batch = []

                    epoch += 1
                    if epoch % MODEL_SAVE_INTERVAL == 0:
                        # Save the neural net parameters to disk.
                        save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                               str(epoch) + ".ckpt")
                        print("Model saved in file: %s" % save_path)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

DRIVE_LEFT = 0 
DRIVE_FORWARD = 1
DRIVE_RIGHT = 2


def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.8
    epsilon_decay = 0.99

    q_table = np.zeros((observation_space.n, action_space.n))

    highest_reward = 0

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table, terminated)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print(f"New highest reward: {total_reward}.")
                saveQ(q_table, "best_q_table.csv")

            env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table, terminated):
    alpha = 0.1
    gamma = 0.9

    #Retrieve the current Q value for the state/action pair
    currentQ = q_table[state][action]

    if terminated:
        target = reward

    else: 
        #Find the maximum Q value for the next state 
        maxNextQ = np.max(q_table[next_state])

        #Calculate target Q value 
        target = reward + gamma*maxNextQ

    #Update original Q value 
    newQ = currentQ + alpha * (target - currentQ)
    q_table[state][action] = newQ

    return q_table


def choose_action(state, q_table, epsilon, action_space):

    """Chooses the next action for the agent. 
    If two states have the same maximum Q value, randomly select the next state."""

    action = 0

    #Randomly generate a number between 0 and 1 to determine if the agent will exploit or explore 
    choose = np.random.rand()

    if (choose < epsilon): 

        maxQ = q_table[state][np.argmax(q_table[state])]
        max_indices = np.where(q_table[state] == maxQ)
        action = np.random.choice(max_indices[0]) #randomly select a number from the array which is the first and only element in the tuple

    else: 
        action = np.random.randint(0, action_space.n)
    
    return action


def saveQ(q_table, save_path):

    np.savetxt(save_path, q_table, delimiter=",")


def loadQ(load_path):

    q_table = np.loadtxt(load_path, delimiter=",")
    return q_table

if __name__ == "__main__":
    main()
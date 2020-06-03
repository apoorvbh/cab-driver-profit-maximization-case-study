# Import routines

import numpy as np
import math
import random

from itertools import product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self, Time_matrix, debug = False):
        """initialise your state and define your action space and state space"""
        
        self.Time_matrix = Time_matrix
        self.debug = debug

        if self.debug:
            print('\nTime Matrix Shape : {}'.format(Time_matrix.shape))

        self.locations = self.allowed_locations()
        self.hours = self.hour_range()
        self.days = self.day_range()

        self.state_size = len(self.locations) + len(self.hours) + len(self.days)

        if self.debug:
            print('\nState Size : {}'.format(self.state_size))

        self.action_space = self.define_action_space()
        self.state_space  = self.define_state_space()

        # Start the first round
        self.reset()

        
    def allowed_locations(self):
        locations = np.arange(1, m+1)

        if self.debug:
            print('\nTotal Locations : {}'.format(len(locations)))

        return locations


    def day_range(self):
        days = np.arange(d)

        if self.debug:
            print('\nTotal no. of days : {}'.format(len(days)))

        return days


    def hour_range(self):
        time = np.arange(t)

        if self.debug:
            print('\nTotal no. of hours : {}'.format(len(time)))

        return time

    def define_state_space(self):
        state_space = list(product(self.locations, self.hours, self.days))

        if self.debug:
            print('\nState Space Size : {}'.format(len(state_space)))

        return state_space


    def define_action_space(self):
        action_space = list(product(self.locations, self.locations))
        action_space = list(filter(lambda x: x[0] != x[1], action_space))

        action_space.append((0,0))

        if self.debug:
            print('\nAction Space Size : {}'.format(len(action_space)))
            print('\nAction Space : {}'.format(action_space))

        return action_space


    def encode_location(self, location):
        encoded_location = np.zeros(m)
        encoded_location[location - 1] = 1

        if self.debug:
            print('\nLocation : {}'.format(location))
            print('Encoded Location : {}'.format(encoded_location))

        return encoded_location


    def encode_day(self, day):
        encoded_day = np.zeros(d)
        encoded_day[day] = 1

        if self.debug:
            print('\nDay : {}'.format(day))
            print('Encoded Day : {}'.format(encoded_day))

        return encoded_day

    
    def encode_time(self, time):
        encoded_time = np.zeros(t)
        encoded_time[time] = 1

        if self.debug:
            print('\nTime : {}'.format(time))
            print('Encoded Time : {}'.format(encoded_time))

        return encoded_time

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.concatenate((self.encode_location(state[0]), self.encode_time(state[1]), self.encode_day(state[2])))

        if self.debug:
            print('\nState : {}'.format(state))
            print('Encoded State : {}'.format(state_encod))

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod

    def get_number_of_requests(self, location):
        location_deman_dict = {1:2, 2:12, 3:4, 4:7, 5:8}

        return np.random.poisson(location_deman_dict.get(location))

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        requests = self.get_number_of_requests(state[0])

        if requests > 15:
            requests = 15

        if self.debug:
            print('\nTotal Requests : {}'.format(requests))

        # (0,0) is not considered as customer request
        possible_actions_index = random.sample(range(0, (m-1)*m), requests) 

        possible_actions_index.append(len(self.action_space) - 1)

        if self.debug:
            print('Possible Actions Index : {}'.format(possible_actions_index))
        
        actions = [self.action_space[i] for i in possible_actions_index]

        if self.debug:
            print('Actions : {}'.format(actions))

        return possible_actions_index, actions   

    def get_time_to_reach(self, state, action):
        if self.debug:
            print('\nCurrent location : {}, Source location : {}, Destination location : {}'.format(state[0], action[0], action[1]))
            print('Current hour of day : {}, Current day of week : {}'.format(state[1], state[2]))

        time_to_reach_source = self.get_time(state[0], action[0], state[1], state[2])

        if self.debug:
            print('Time to reach source : {}'.format(time_to_reach_source))

        updated_hour_of_day, updated_day_of_week = self.get_updated_time(state[1], state[2], time_to_reach_source)

        if self.debug:
            print('Updated hour of day : {}, Updated day of week : {}'.format(updated_hour_of_day, updated_day_of_week))

        time_to_reach_dest = self.get_time(action[0], action[1], updated_hour_of_day, updated_day_of_week)

        if self.debug:
            print('Time to reach destination : {}'.format(time_to_reach_dest))

        return time_to_reach_source, time_to_reach_dest

    
    def get_updated_time(self, current_hour_of_day, current_day_of_week, time_taken):
        updated_time = current_hour_of_day + time_taken
        updated_hour_of_day = (updated_time % 24)
        updated_day_of_week = current_day_of_week + (updated_time // 24)

        if updated_day_of_week > 6:
            updated_day_of_week = 0

        return updated_hour_of_day, updated_day_of_week
    
    
    def reward_func(self, time_to_reach_source, time_to_reach_dest):
        """Takes in state, action and Time-matrix and returns the reward"""
        reward = R*time_to_reach_dest - C*(time_to_reach_dest + time_to_reach_source)

        if self.debug:
            print('\nPer Hour Revenue : {}'.format(R))
            print('Per Hour Cost : {}'.format(C))

        return reward


    def next_state_func(self, state, next_location, time_taken):
        """Takes state and action as input and returns next state"""
        updated_hour_of_day, updated_day_of_week = self.get_updated_time(state[1], state[2], time_taken)
        
        next_state = [next_location, updated_hour_of_day, updated_day_of_week]
        
        return next_state


    def get_time(self, start_location, end_location, time_of_day, day):
        if self.debug:
            print('\nStart Location : {}'.format(start_location))
            print('End Location : {}'.format(end_location))
            print('Hour : {}'.format(time_of_day))
            print('Day : {}'.format(day))
        
        if (start_location > 0) and (end_location > 0):
            time_taken = int(self.Time_matrix[start_location - 1][end_location - 1][time_of_day][day])
        else:
            time_taken = 0
            
        return time_taken

    
    def step(self, state, action):
        if self.debug:
            print('\nCurrent State : {}'.format(state))
            print('Current Action : {}'.format(action))

        if action == (0,0):
            time_to_reach_source = 1
            time_to_reach_dest = 0
            next_location = state[0]
        else:
            next_location = action[1]
            time_to_reach_source, time_to_reach_dest = self.get_time_to_reach(state, action)

        time_taken = time_to_reach_source + time_to_reach_dest
        next_state = self.next_state_func(state, next_location, time_taken)
        reward = self.reward_func(time_to_reach_source, time_to_reach_dest)

        if self.debug:
            print('\nTotal Time Taken : {}'.format(time_taken))
            print('Next State : {}'.format(next_state))
            print('Reward : {}'.format(reward))

        return next_state, reward, time_taken


    def reset(self):
        self.state_init = random.choice(self.state_space)
        return self.state_init

    
    def test_env(self):
        state = random.choice(self.state_space)
        self.state_encod_arch1(state)

        actions = self.requests(state)[1]
        selected_action = random.choice(actions)
    
        self.step(state, selected_action)

        
    
if __name__ == "__main__":

    Time_matrix = np.load('TM.npy')

    env = CabDriver(Time_matrix, debug= True)

    env.test_env()

    
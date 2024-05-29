import numpy as np
import matplotlib.pyplot as plt

class Dynamics:
    row = 5
    col = 5
    policy = np.full((row, col),' ', dtype='U{}'.format(len('  ')))
    episode_count = 50
    theta = 0.0001
    gamma = 0.9
    alpha = 0.8
    actions = ["au","ad","al","ar"]
    stochastic_prob = {"right_direction": 0.8, "veer_right":0.05, "veer_left":0.05, "break":0.1}
    obstacle_states = ["22","32"]
    water_states = ["42"]
    goal_state = ["44"]
    
    optimal_policy = np.array([['ar', 'ar', 'ar', 'ad', 'ad'],
                    ['ar', 'ar', 'ar', 'ad', 'ad'],
                    ['au', 'au', 'NA', 'ad', 'ad'],
                    ['au', 'au', 'NA', 'ad', 'ad'],
                    ['au', 'au', 'ar', 'ar', 'G']])
    
    optimal_value = np.array([[4.0187, 4.5548, 5.1575, 5.8336, 6.4553 ],
                            [4.3716, 5.0324, 5.8013, 6.6473, 7.3907 ],
                            [3.8672, 4.39,  0.0000, 7.5769, 8.4637 ],
                            [3.4182, 3.8319,  0.0000, 8.5738, 9.6946 ],
                            [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]])
    
    transition_probabilities = {'00': {'au': {'01': 0.05, '00': 0.95},
        'ad': {'10': 0.8, '01': 0.05, '00': 0.15},
        'ar': {'01': 0.8, '10': 0.05, '00': 0.15},
        'al': {'10': 0.05, '00': 0.95}},

'01': {'au': {'02': 0.05, '00': 0.05, '01': 0.9},
        'ad': {'11': 0.8, '02': 0.05, '00': 0.05, '01': 0.1},
        'ar': {'02': 0.8, '11': 0.05, '01': 0.15},
        'al': {'00': 0.8, '11': 0.05, '01': 0.15}},

'02': {'au': {'03': 0.05, '01': 0.05, '02': 0.9},
        'ad': {'12': 0.8, '03': 0.05, '01': 0.05, '02': 0.1},
        'ar': {'03': 0.8, '12': 0.05, '02': 0.15},
        'al': {'01': 0.8, '12': 0.05, '02': 0.15}},

'03': {'au': {'04': 0.05, '02': 0.05, '03': 0.9},
        'ad': {'13': 0.8, '04': 0.05, '02': 0.05, '03': 0.1},
        'ar': {'04': 0.8, '13': 0.05, '03': 0.15},
        'al': {'02': 0.8, '13': 0.05, '03': 0.15}},

'04': {'au': {'03': 0.05, '04': 0.95},
        'ad': {'14': 0.8, '03': 0.05, '04': 0.15,},
        'ar': {'14': 0.05, '04': 0.95},
        'al': {'03': 0.8, '14': 0.05, '04': 0.15}},

'10': {'au': {'00': 0.8, '11': 0.05, '10': 0.15},
        'ad': {'20': 0.8, '11': 0.05, '10': 0.15},
        'ar': {'11': 0.8, '00': 0.05, '20': 0.05, '10': 0.1},
        'al': {'10': 0.9, '00': 0.05, '20': 0.05}},

'11': {'au': {'01': 0.8, '12': 0.05, '10': 0.05, '11': 0.1},
        'ad': {'21': 0.8, '12': 0.05, '10': 0.05, '11': 0.1},
        'ar': {'12': 0.8, '01': 0.05, '21': 0.05, '11': 0.1},
        'al': {'10': 0.8, '01': 0.05, '21': 0.05, '11': 0.1}},

'12': {'au': {'02': 0.8, '13': 0.05, '11': 0.05, '12': 0.1},
        'ad': {'13': 0.05, '11': 0.05,'12': 0.9},
        'ar': {'13': 0.8, '02': 0.05, '12': 0.15},
        'al': {'11': 0.8, '02': 0.05, '12': 0.15}},


'13': {'au': {'03': 0.8, '14': 0.05, '12': 0.05, '13': 0.1},
        'ad': {'23': 0.8, '14': 0.05, '12': 0.05, '13': 0.1},
        'ar': {'14': 0.8, '03': 0.05, '23': 0.05, '13': 0.1},
        'al': {'12': 0.8, '03': 0.05, '23': 0.05, '13': 0.1}},

'14': {'au': {'04': 0.8, '13': 0.05, '14': 0.15},
        'ad': {'24': 0.8, '13': 0.05, '14': 0.15},
        'ar': {'04': 0.05, '24': 0.05, '14': 0.9},
        'al': {'13': 0.8, '04': 0.05, '24': 0.05, '14': 0.1}},

'20': {'au': {'10': 0.8, '21': 0.05, '20': 0.15},
        'ad': {'30': 0.8, '21': 0.05, '20': 0.15},
        'ar': {'21': 0.8, '10': 0.05, '30': 0.05, '20': 0.1},
        'al': {'10': 0.05, '30': 0.05, '20': 0.9}},

'21': {'au': {'11': 0.8, '20': 0.05, '21': 0.15},
        'ad': {'31': 0.8, '20': 0.05, '21': 0.15},
        'ar': {'11': 0.05, '31': 0.05, '21': 0.9},
        'al': {'20': 0.8, '11': 0.05, '31': 0.05, '21': 0.1}},

'22': {'au': {'12': 0.8, '23': 0.05, '21': 0.05, '22': 0.1},
        'ad': {'23': 0.05, '21': 0.05, '22': 0.9},
        'ar': {'23': 0.8, '12': 0.05, '22': 0.15},
        'al': {'21': 0.8, '12': 0.05, '22': 0.15}},

'23': {'au': {'13': 0.8, '24': 0.05, '23': 0.15},
        'ad': {'33': 0.8, '24': 0.05, '23': 0.15},
        'ar': {'24': 0.8, '13': 0.05, '33': 0.05, '23': 0.1},
        'al': {'13': 0.05, '33': 0.05, '23': 0.9}},

'24': {'au': {'14': 0.8, '23': 0.05, '24': 0.15},
        'ad': {'34': 0.8, '23': 0.05, '24': 0.15},
        'ar': {'14': 0.05, '34': 0.05, '24': 0.9},
        'al': {'23': 0.8, '14': 0.05, '34': 0.05, '24': 0.1}},

'30': {'au': {'20': 0.8, '31': 0.05, '30': 0.15},
        'ad': {'40': 0.8, '31': 0.05, '30': 0.15},
        'ar': {'31': 0.8, '20': 0.05, '40': 0.05, '30': 0.1},
        'al': {'20': 0.05, '40': 0.05, '30': 0.9}},

'31': {'au': {'21': 0.8, '30': 0.05, '31': 0.15},
        'ad': {'41': 0.8, '30': 0.05, '31': 0.15},
        'ar': {'21': 0.05, '41': 0.05, '31': 0.9},
        'al': {'30': 0.8, '21': 0.05, '41': 0.05, '31': 0.1}},


'32': {'au': {'33': 0.05, '31': 0.05, '32': 0.9},
        'ad': {'42': 0.8, '33': 0.05, '31': 0.05, '32': 0.1},
        'ar': {'33': 0.8, '42': 0.05, '32': 0.15},
        'al': {'31': 0.8, '42': 0.05, '32': 0.15}},

'33': {'au': {'23': 0.8, '34': 0.05, '33': 0.15},
        'ad': {'43': 0.8, '34': 0.05, '33': 0.15},
        'ar': {'34': 0.8, '23': 0.05, '43': 0.05, '33': 0.1},
        'al': {'23': 0.05, '43': 0.05, '33': 0.9}},


'34': {'au': {'24': 0.8, '33': 0.05, '34': 0.15},
        'ad': {'44': 0.8, '33': 0.05, '34': 0.15},
        'ar': {'24': 0.05, '44': 0.05, '34': 0.9},
        'al': {'33': 0.8, '24': 0.05, '44': 0.05, '34': 0.1}},

'40': {'au': {'30': 0.8, '41': 0.05, '40': 0.15},
        'ad': {'41': 0.05, '40': 0.95},
        'ar': {'41': 0.8, '30': 0.05, '40': 0.15},
        'al': {'30': 0.05, '40': 0.95}},

'41': {'au': {'31': 0.8, '42': 0.05, '40': 0.05, '41': 0.1},
        'ad': {'42': 0.05, '40': 0.05, '41': 0.9},
        'ar': {'42': 0.8, '31': 0.05, '41': 0.15},
        'al': {'40': 0.8, '31': 0.05, '41': 0.15}},

'42': {'au': {'43': 0.05, '41': 0.05, '42': 0.9},
        'ad': {'43': 0.05, '41': 0.05, '42': 0.9},
        'ar': {'43': 0.8, '42': 0.2},
        'al': {'41': 0.8, '42': 0.2}},
        
'43': {'au': {'33': 0.8, '44': 0.05, '42': 0.05, '43': 0.1},
        'ad': {'44': 0.05, '42': 0.05, '43': 0.9},
        'ar': {'44': 0.8, '33': 0.05, '43': 0.15},
        'al': {'42': 0.8, '33': 0.05, '43': 0.15}},

'44': {'au': {'34': 0.8, '44': 0.15, '43': 0.05},
        'ad': {'43': 0.05, '44': 0.95},
        'ar': {'34': 0.05, '44': 0.95},
        'al': {'43': 0.8, '34': 0.05, '44': 0.15}}}


class TDLearning:   
    def get_initial_values():
        initial_state = np.zeros((Dynamics.row, Dynamics.col))
        initial_state[2][2] = 0
        initial_state[3][2] = 0
        initial_state[4][4] = 0
        return initial_state   
    
    def run_Grid_World(initial_state, value_state):   
        episode_count = 0 
        reward = 0   
        while True:              
        #     print(f"Episode:{episode_count}")
            state = initial_state
            curr_state = state
            new_value_state = value_state
            while True:                
                # print(f"Initial state : {curr_state}")
                # print(count)
                if curr_state in Dynamics.goal_state:
                        break
                curr_r, curr_c = [int(char) for char in curr_state]
                action = Dynamics.optimal_policy[curr_r][curr_c]
                available_next_states = list(Dynamics.transition_probabilities[curr_state][action].keys())
                next_state_probabilities = list(Dynamics.transition_probabilities[curr_state][action].values())
                next_state = np.random.choice(available_next_states, p=next_state_probabilities)  
                r, c = [int(char) for char in next_state]
                next_state_value = value_state[r][c]
                if(next_state in Dynamics.water_states):
                    reward = -10
                if(next_state in Dynamics.goal_state):
                    reward = 10
                episode_count = 25.3
                new_value_state[curr_r][curr_c] = new_value_state[curr_r][curr_c] + Dynamics.alpha * (reward + (Dynamics.gamma * next_state_value) - new_value_state[curr_r][curr_c])
                curr_state = next_state
                # print(f"next state : {curr_state}")            
        #     print(count)
            episode_count = episode_count+1
            max_norm = np.max(np.abs(new_value_state - value_state))
            value_state = new_value_state
            if max_norm < Dynamics.theta:
                break
        return value_state, episode_count
                
def main():   
        value_state = TDLearning.get_initial_values()     
        total_values = np.zeros((Dynamics.row, Dynamics.col))
        total_episodes = 0
        episodes_list = []
        for i in range(Dynamics.episode_count):
                initial_state_list = [key for key, value in Dynamics.transition_probabilities.items() if key not in Dynamics.obstacle_states]
                selected_state = np.random.choice(initial_state_list)               
                final_value_state,count = TDLearning.run_Grid_World(selected_state, value_state)  
                total_values += final_value_state
                total_episodes += count
                episodes_list.append(count)
        average_values = total_values/Dynamics.episode_count
        average_episodes = total_episodes/Dynamics.episode_count
        max_norm = np.max(np.abs(average_values - Dynamics.optimal_value))        
        std_dev = np.std(episodes_list)
        print("\n\n")
        print(f"Max norm : {round(max_norm,4)}")
        print("\n\n")
        print(f"Average episodes : {round(average_episodes,4)}")
        print("\n\n")
        print(f"Standard Deviation : {round(std_dev,4)}")
        print("\n\n")
        print(f"Average value function:\n")
        for row in average_values:
            for value in row:
                print(round(value,4).astype(float), end="\t\t")
            print()
        print("\n\n")
        return max_norm, average_episodes, average_values


if __name__ == "__main__":
    main()
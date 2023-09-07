import Enviroment as env
import numpy as np

# Number of APs
NUM_OF_AP = 1
# Number of Devices
NUM_OF_DEVICE = 3 
# Maximum Packet Loss Rate 
RHO_MAX = 0.1
# L_k
# Number of Q-tables
I = 1

# PLOT DATA POINTS

# CREAT STATE 
# State is a NUM_OF_DEVICE*4 matrix 
# where state[k]=[QoS_satisfaction_sub, QoS_satisfaction_mW, ACK feedback at t-1 on sub, ACK feedback at t-1 on mW]
def initialize_state():
    state = np.matrix(np.zeros(shape=(NUM_OF_DEVICE,4)))
    return state
# packet_loss_rate is a vector 
# in which packet_loss_rate[0] is the packet loss rate on Sub6-GHz interface, packet_loss_rate[1] is the packet loss rate on milimeter-Wave interface
def update_state(state,packet_loss_rate,num_of_success_packet):
    for k in range (NUM_OF_DEVICE):
        for i in range (2):
            #QoS satisfaction
            if(packet_loss_rate[i]<RHO_MAX):
                state[k,i]=1
            elif(packet_loss_rate[i]>=RHO_MAX):
                state[k,i]=0
            #Number of successfully delivered packet on each interface
            state[k,i+2]=num_of_success_packet[i]
    return state

# CREATE ACTION
def chose_action():
    
# CREATE REWARD
def reward(state,action,frame_num):
    
# CREATE MODEL
    # A Q-table is a dictionary with key=tuple(state.apend(action)) 
    def initialize_Q_tables():
        Q_tables = []
        for i in range(I):
            Q={}
            Q_tables.append(Q)
        return Q_tables
    
# TRAINING
    # Read from old Q-tables

    #
import Enviroment as env
import numpy as np

# Number of APs
NUM_OF_AP = 1
# Number of Devices
NUM_OF_DEVICE = 3 
# Number of Sub-6GHz channels, Number of MmWave beams
N = M = 4
# Maximum Packet Loss Rate 
RHO_MAX = 0.1
# L_k
L_k = 6
# Number of Frame
T = 10000
# Risk control
LAMBDA_P = 0.5
# Ultility function paremeter
BETA = -0.5
# Learning parameters
GAMMA = 0.9
EPSILON = 0.5
# Decay factor
LAMBDA = 0.995
# p_max
P_MAX = 0.1
# Number of Q-tables
I = 2
X0 = -1

# PLOT DATA POINTS

# CREAT STATE 
# State is a NUM_OF_DEVICE*4 matrix 
# where state[k]=[QoS_satisfaction_sub, QoS_satisfaction_mW, ACK feedback at t-1 on sub, ACK feedback at t-1 on mW]
def initialize_state():
    state = np.matrix(np.zeros(shape=(NUM_OF_DEVICE,4)))
    return state

# packet_loss_rate is a vector 
# in which packet_loss_rate = [packet loss rate on Sub6-GHz, packet loss rate on milimeter-Wave]
def update_state(state,packet_loss_rate,feedback):
    for k in range (NUM_OF_DEVICE):
        for i in range (2):
            #QoS satisfaction
            if(packet_loss_rate[i]<RHO_MAX):
                state[k,i]=1
            elif(packet_loss_rate[i]>=RHO_MAX):
                state[k,i]=0
            #Number of successfully delivered packet on each interface
            state[k,i+2] = feedback[i]
    return state

# CREATE ACTION
# Action is an array where action[k] is the action of device k 
def initialize_action():
    # Initialize random action at the beginning
    action = np.random.randint(0,3,NUM_OF_DEVICE)
    return action

# Check if the bandwidth is full, return decent action
def check_action(action):
    sub = [] # Containts index of devices on sub
    mW = [] # Containts index of devices on mW
    sub_num = 0
    mW_num = 0
    for a in action:
        match a:
            case 0:
                sub.append(a)
                sub_num+=1
            case 1:
                mW.append(a)
                mW_num+=1
            case 2:
                sub.append(a)
                mW.append(a)
                sub_num+=1
                mW_num+=1

    while(sub_num>N):
        # Change random device
        change_index = np.random.randint(0,len(sub)-1)
        if(mW_num<M):
            action[change_index] = 1
        sub.remove(change_index)

    while(mW_num):
        change_index = np.random.randint(0,len(sub)-1)
        if(sub_num<N):
            action[change_index] = 0      
        mW.remove(change_index)

    return action    
      
def chose_action(state,Q_table):
    # Epsilon-Greedy
    p=np.random.rand()
    sub = 0
    mW = 0
    if(p<EPSILON):
        action = initialize_action()
        action = check_action(action)
        return action
    else:
        max_Q = -np.Infinity
        for i in Q_table:
            state_i = tuple(map(tuple,i[:,0:4]))
            if(state_i == state & Q_table[i]>max_Q):
                max_Q = Q_table[i]
                i = np.array(i)
                action = i[:,4]
        action = check_action(action)
        return action
    
# Return an matrix where number_of_packet[k] = [number of transmit packets on sub, number of transmit packets on mW] 
def perform_action(action,l_sub_max,l_mW_max):
    number_of_packet = np.matrix(np.zeros(shape=(NUM_OF_DEVICE,2)))
    for k in range(NUM_OF_DEVICE):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        if(action==0):
            number_of_packet[k,0] = min(l_sub_max_k,L_k)
            number_of_packet[k,1] = 0
        
        if(action==1):
            number_of_packet[k,0] = 0
            number_of_packet[k,1] = min(l_mW_max_k,L_k)
        
        if(action==2):
            if(l_mW_max_k < L_k):
                number_of_packet[k,1] = l_mW_max_k
                number_of_packet[k,0] = min(l_sub_max_k, L_k - l_mW_max_k)
            if(l_mW_max_k >= L_k):
                number_of_packet[k,0] = 1
                number_of_packet[k,1] = L_k - 1
    return number_of_packet

# Return an matrix where feedback[k] = [number of received packets on sub, number of received packets on mW] 
def receive_feedback(l_sub_max,l_mW_max):
    feedback = np.matrix(np.zeros(shape=(NUM_OF_DEVICE,2)))
    action_performed = perform_action(action,l_sub_max,l_mW_max)
    for k in range(NUM_OF_DEVICE):
        l_sub = action_performed[k,0]
        l_mW = action_performed[k,1]

        feedback[k,0] = min(l_sub,l_sub_max)
        feedback[k,1] = min(l_mW,l_mW_max)

    return feedback

           
def compute_packet_loss_rate(frame_num,old_packet_loss_rate,feedback,send_packet_num):
    packet_loss_rate = np.matrix(np.zeros(shape=(NUM_OF_DEVICE,2)))
    for k in range(NUM_OF_DEVICE):
        # Packet Successfull Rate of device k over Sub-6GHz Interface
        packet_successfull_rate_sub = env.packet_successful_rate(feedback[k,0])
        # Packet Successfull Rate of device k over MmWave Interface
        packet_successfull_rate_mW = env.packet_successful_rate(feedback[k,1])
        
        l_sub_k = send_packet_num[k,0]
        l_mW_k = send_packet_num[k,1]
        
        packet_loss_rate[k,0]= env.packet_loss_rate(frame_num,old_packet_loss_rate[k,0],packet_successfull_rate_sub,l_sub_k)
        packet_loss_rate[k,0]= env.packet_loss_rate(frame_num,old_packet_loss_rate[k,1],packet_successfull_rate_mW,l_mW_k)

    return packet_loss_rate

# CREATE REWARD
# Initialize a reward table as a dictionary, 
def initialize_reward(state,action):
    reward = {}
    return reward
    
# Update reward table in case it did not have the (state,action) 
def update_reward(state,action,old_reward):
    state_action = tuple (state.append(action))
    if(not(state_action in old_reward)):
        old_reward.update({state_action:0})
    return old_reward

def compute_reward(state,action,old_reward,omega_sub,omega_mW,frame_num):
    sum = 0
    for k in range(NUM_OF_DEVICE):
        state_k = state[k]
        action_k = action[k]
        l_k = perform_action(action)
        sum = sum + (omega_sub[k] + omega_mW[k])/(l_k[0]+l_k[1]) - (1- state_k[0]) - (1-state_k[1])
    
    sum = (((frame_num - 1)*old_reward) + sum)/frame_num 
    return sum

    
# CREATE MODEL
# A Q-table is a dictionary with key=tuple(state.apend(action)), value = Q(state,action)
def initialize_Q_tables():
    Q_tables = []
    for i in range(I):
        Q={}
        Q_tables.append(Q)
    return Q_tables

def add_2_Q_tables(Q1,Q2):
    for item in Q2:
        if(item in Q1):
            Q1[item] += Q2[item]
        else:
            Q1.update({item:Q2[item]})
    return Q1

def adverage_Q_table(Q_tables):
    for i in range(len(Q_tables)):
        res = add_2_Q_tables (res,Q_tables[i])
    for i in res:
        res[i]/=I
    return res

def compute_risk_adverse_Q(Q_tables, random_Q_index):
    Q_random = Q_tables[random_Q_index]
    Q_adverage = adverage_Q_table(Q_tables)
    sum_sqrt = {}
    for i in range(I):
        minus_Q_adverage = {}
        for j in Q_adverage:
            minus_Q_adverage.update({j,-Q_adverage[j]})
        
        sub = add_2_Q_tables(Q_tables[i],minus_Q_adverage)

        for j in sub:
            sub[j]*=sub[j]

    sum_sqrt = add_2_Q_tables(sum_sqrt,sub)
    
    for i in sum_sqrt:
        sum_sqrt[i] = sum_sqrt[i]*LAMBDA_P/(I-1)

    res = add_2_Q_tables(Q_random,sum_sqrt)
    return res

def u(x):
    return -np.exp(BETA*x)

def update_Q_table(Q_table,alpha,next_state):
    for i in Q_table:
        state = np.asarray(i[0:4])
        action = i[4]

        # Find max(Q(s(t+1),a)
        max_Q = -np.Infinity
        for i in Q_table:
            state_i = np.asarray(i[0:4])
            if((state_i == next_state) & (Q_table[i]>max_Q)):
                max_Q = Q_table[i]

        Q_table[i] = Q_table[i] + alpha[i]*(u( compute_reward(state,action) + GAMMA * max_Q - Q_table[i] ) - X0)
    
    return Q_table

def initialize_V():
    return initialize_Q_tables()

def update_V(V,Q_table):
    for i in Q_table:
        if(i in V):
            V[i] += 1
        else:
            V.update({i:1})
    return V

def initialize_alpha():
    return initialize_Q_tables()

def update_alpha(alpha,V):
    for i in V:
        if(i in alpha):
            alpha[i] = 1/V[i]
        else:
            alpha.update({i:1/V[i]})
    return alpha

# Set up environment
# Complex channel coefficient
def generate_h_tidle(mu,sigma):
    h_tidle = []
    h_tidle_sub = np.empty(shape=(NUM_OF_DEVICE,env.NUM_OF_SUB_CHANNEL),dtype=complex)
    h_tidle_sub[:,:] = env.generate_h_tilde(mu,sigma)

    h_tidle_mW= np.empty(shape=(NUM_OF_DEVICE,env.NUM_OF_BEAM),dtype=complex)
    h_tidle_mW[:,:] = env.generate_h_tilde(mu,sigma)

    h_tidle.append(h_tidle_sub)
    h_tidle.append(h_tidle_mW)
    return h_tidle

# Achievable rate 
# r[0] = r_sub, r[1] = r_mW for all device k, subchannel n, beam m
def compute_r(device_positions,h_tidle):
    r = []
    r_sub = np.empty(shape=(NUM_OF_DEVICE,env.NUM_OF_SUB_CHANNEL))
    r_mW = np.empty(shape=(NUM_OF_DEVICE,env.NUM_OF_BEAM))
    h_tidle_sub = h_tidle[0]
    h_tidle_mW = h_tidle[1]

    h_sub = env.compute_devices_h_sub(list_of_devices=device_positions,h_tilde=h_tidle_sub)
    h_mW = env.compute_devices_h_sub(list_of_devices=device_positions,h_tilde=h_tidle_mW)

    for k in range(NUM_OF_DEVICE):
        for n in range(env.NUM_OF_SUB_CHANNEL):
            r_sub[k,n] = env.r_sub(h_sub,0,k,N)
        for m in range(env.NUM_OF_BEAM):
            r_mW[k,m] = env.r_mW(h_mW,0,k,m,eta=,beta=)

    r.append(r_sub)
    r.append(r_mW)
    return r

# l_max[k] = [l_sub_max,l_mW_max]
def compute_l_max(r):
    return r*env.T/env.D

    
# TRAINING
# Read from old Q-tables

# Train with new data
device_postions = env.initialize_devices_pos()
state = initialize_state()
Q_tables = initialize_Q_tables()
V = initialize_V()
alpha = initialize_alpha()
packet_loss_rate = np.matrix(np.zeros(shape=(NUM_OF_DEVICE,2)))

for frame in range(T):
    # Random Q-table
    H = np.random.randint(0,I)
    risk_adverse_Q = compute_risk_adverse_Q(Q_tables,H)
    
    # Update EPSILON
    EPSILON = EPSILON * LAMBDA

    # Set up environment
    h_tidle = env.generate_h_tilde(0,1)
    r = compute_r(device_postions,h_tidle)
    l_max = compute_l_max(r)
    l_sub_max = l_max[0][:,0]
    l_mW_max = l_max[1][:,1]

    # Select action
    action = chose_action(risk_adverse_Q)

    # Perform action and get ACK/NACK feedback
    number_of_send_packet = perform_action(action)
    feedback = receive_feedback(l_sub_max,l_mW_max)
    packet_loss_rate = compute_packet_loss_rate(frame,packet_loss_rate,feedback,number_of_send_packet)

    # Compute reward
    if(frame == 1):
        reward = compute_reward(state,action,0)
    else:
        reward = compute_reward(state,action,reward)

    # Generate mask J
    J = np.random.poisson(1,I)

    for i in range(I):
        if(J[i]==1):
            Q_table = update_Q_table(Q_tables[i],alpha[i])
            V[i] = update_V(V[i],Q_table)
            alpha[i] = update_alpha(alpha[i],V[i])

    state = update_state(state,)
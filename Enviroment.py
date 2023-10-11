import numpy as np
import random as rd
import matplotlib.pyplot as plt

#the considered space has a width of 90 meters, a length of 90 meters
length = 90
width = 90

length_of_cell = 0.001
number_of_area_per_row = 3
length_of_area = length/number_of_area_per_row

#Number of APs
NUM_OF_AP = 1
#Number of Devices
NUM_OF_DEVICE = 3
#Number of Sub-6Ghz channels N and mmWave beam M
NUM_OF_SUB_CHANNEL = 3
NUM_OF_BEAM = 3
#Transmit Power P_sub = P_mW = P ~ 5dBm 
P = pow(10,5/10)
#Noise Power sigma^2 ~ -169dBm/Hz
SIGMA_SQR = pow(10,-169/10)
#Bandwidth W = 10MHz
W = 10000000
#Frame Duration T_s
T = 1
#Packet size D = 8 bit
D = 8

#initialize position of AP.
#the AP was located at the central of the area
#the position of each AP is the constant
AP_POSITION=(45,45)

#the function calculates the distance to the nearest AP
def distance_to_AP(pos_of_device):
    distance = np.sqrt((pos_of_device[0] - AP_POSITION[0]) * (pos_of_device[0] - AP_POSITION[0]) + (pos_of_device[1] - AP_POSITION[1]) * (pos_of_device[1] - AP_POSITION[1]))
    return distance


#initialize device's postion with random value
#after initializing any device's position, check the distance from that device to the nearest AP,
#if the distance is satisfied, store it into the array list_of_devices.
def initialize_devices_pos():
    list_of_devices = []
    i = 0
    while i < NUM_OF_DEVICE:
        list_of_devices.append((rd.uniform(0,length),rd.uniform(0,width)))
        if(distance_to_AP(pos_of_device= list_of_devices[i]) >= 0.001):
            i = i+1
        else:
            list_of_devices.remove(list_of_devices[i])
    return list_of_devices

#The list contains positions of devices
list_of_devices=initialize_devices_pos()

#Path loss model
def path_loss_sub(distance):
    return 38.5 + 30*(np.log10(distance))
def path_loss_mW_los(distance):
    X=np.random.normal(0,5.8)
    return 61.4 + 20*(np.log10(distance))+X
def path_loss_mw_nlos(distance):
    X=np.random.normal(0,8.7)
    return 72 + 29.2*(np.log10(distance))+X


# Main Transmit Beam Gain G_b
def G(eta,beta):
    epsilon=0.1
    return (2*np.pi-(2*np.pi-eta)*epsilon)/(eta)


#Channel coefficient h=h_tilde* 10^(-pathloss/20)
#h_tilde = (a + b*i)/sqrt(2)
#in which a and b is random value from a Normal distribution
def generate_h_tilde(mu,sigma):
    re=np.random.normal(mu,sigma,1)[0]
    im=np.random.normal(mu,sigma,1)[0]
    h_tilde=complex(re,im)/np.sqrt(2)
    return h_tilde

def compute_h_sub(list_of_devices,device_index,h_tilde):
    h=h_tilde* pow(10,-path_loss_sub(distance_to_AP(list_of_devices[device_index]))/20.0)
    return h

def compute_h_mW(list_of_devices,device_index,eta,beta,h_tilde):
    path_loss
    #device blocked by obstacle
    if(device_index==2 or device_index==6):
        path_loss=path_loss_mw_nlos(distance_to_AP(list_of_devices[device_index]))
    #device not blocked
    else:
        path_loss=path_loss_mW_los(distance_to_AP(list_of_devices[device_index]))
    h=G(eta,beta)*h_tilde*pow(10,-path_loss/20)*G_Rxk
    return h


#return a matrix of channel coefficient h between device k and AP on subchannel n
def compute_devices_h_sub(list_of_devices,h_tilde):
    list_of_devices_h=np.matrix(np.zeros([NUM_OF_DEVICE,NUM_OF_SUB_CHANNEL]))
    for k in range(NUM_OF_DEVICE):
        for n in range (NUM_OF_SUB_CHANNEL):
            list_of_devices_h[k,n]=compute_h_sub(list_of_devices,k,h_tilde)
    return list_of_devices_h
#return a matrix of channel coefficient h between device k and AP on beam m
def compute_devices_h_mW(list_of_devices,eta,beta,h_tilde):
    list_of_devices_h=np.matrix(np.zeros([NUM_OF_DEVICE,NUM_OF_SUB_CHANNEL]))
    for k in range(NUM_OF_DEVICE):
        for m in range (NUM_OF_BEAM):
            list_of_devices_h[k,m]=compute_h_mW(list_of_devices,k,eta,beta,h_tilde)
    return list_of_devices_h


# gamma_sub(h,k,n) (t) is the Signal to Interference-plus-Noise Ratio (SINR) from AP to device k on subchannel n with channel coefficient h
def gamma_sub(h,device_index,channel_index):
    power=pow(abs(h[device_index,channel_index]),2)*P
    interference_plus_noise=W*SIGMA_SQR
    for b in range(NUM_OF_DEVICE):
        if(b!=device_index):
            interference_plus_noise += pow(abs(h[device_index,channel_index]),2)*P
    return power/interference_plus_noise
# gamma_mW(k,m) (t) is the Signal to Interference-plus-Noise Ratio (SINR) from AP to device k on beam m with channel coeffiction h
def gamma_mW(h,device_index,beam_index):
    power=pow(abs(h[device_index,beam_index]),2)*P
    interference_plus_noise=W*SIGMA_SQR
    for b in range(NUM_OF_DEVICE):
        if(b!=device_index):
            interference_plus_noise += pow(abs(h[device_index,beam_index]),2)*P
    return power/interference_plus_noise


#achievable data rate r_bkf (t) for the link between
# AP b, device k and for application f using bandwidth Wf at scheduling frame t
def r_sub(h,AP_index,device_index,channel_index):
    return W*np.log2(1+gamma_sub(h,device_index,channel_index))
def r_mW(h,AP_index,device_index,beam_index,eta,beta):
    return W*np.log2(1+gamma_sub(h,device_index,beam_index))


# Maximum number of packets of size d bits that can be received successfully at device k on interface v
def l_max(r_bkv):
    return r_bkv*T/D

# Number of sucessfully received packets at device k on interface v 
def num_of_success_packets(l_kv,l_kv_max):
    return l_kv-l_kv_max
# Packet Successful Delivery rate at device k on interface v 
def packet_successful_rate(num_of_success_packet,num_of_packet):
    return num_of_success_packet/num_of_packet

def packet_loss_rate(t,old_packet_loss_rate,packet_successful_rate,l_kv):
    if(l_kv==0):
        packet_loss_rate=((t-1)/t)*old_packet_loss_rate
        return packet_loss_rate
    elif(l_kv>0):
        packet_loss_rate=(1/t)*((t-1)*old_packet_loss_rate + (1-packet_successful_rate))
        return packet_loss_rate

#Plot APs and devices Position
plt.title("AP and devices Position")
device_x,device_y=zip(*list_of_devices)
plt.scatter(AP_POSITION[0],AP_POSITION[1],cmap='hot')
plt.scatter(device_x,device_y,cmap='hot')
plt.grid()
plt.show(block=False)


#the value of r_bkf is immediate
#suppose in the real world, we have 10000 frames that devices have to take action
#suppose the transmit power not depend on application f of device k -> r_bkf depends on which device k of AP b is?
#each frame has its r_bkf 
#Simulating 10000 frames, determine the value of r_bkf in each frame
# def write_data(list_of_r_from_0_to_t,NUM_OF_FRAME):
#     list_of_r_from_0_to_t = [[] for i in range(NUM_OF_FRAME)]
#     file = open("data_r.txt", "w")

#     for i in range(NUM_OF_FRAME):
#         file.writelines(' FRAME START! '.center(200,'=')+"\n")
#         for b in range(NUM_OF_AP):
#             for k in range(NUM_OF_DEVICE):
#                 r_bkf = r(b, k, application_index=1)
#                 file.write(f"{str(round(r_bkf,5)): <20}")
#                 list_of_r_from_0_to_t[i].append(r_bkf)
#             file.writelines("\n")
#         file.writelines(' END OF FRAME! '.center(200,'=')+"\n\n")

#     file.close()

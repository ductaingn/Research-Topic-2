import numpy as np
import random as rd
import matplotlib.pyplot as plt

#the considered space has a width of 0.9 kilometers, a length of 0.9 kilometers
length = 0.09
width = 0.09

length_of_cell = 0.001
number_of_area_per_row = 3
length_of_area = length/number_of_area_per_row

#Number of APs
NUM_OF_AP = 9
#Number of Users
NUM_OF_USER = 10
#Transmit Power P ~ 5dBm 
P = pow(10,5/10)
#Noise Power sigma^2 ~ -169dBm/Hz
SIGMA_SQR = pow(10,-169/10)
#Bandwidth W = 10MHz
W = 10000000

#The list contains positions of APs
list_of_AP = []

#initialize position of each AP.
#each AP was located at the central of each area
#the position of each AP is the constant
for i in range(NUM_OF_AP): 
    list_of_AP.append((i % 3 * length_of_area + length_of_area / 2 , (i - i%3) / 3 * length_of_area + length_of_area / 2))

#the function calculates the distance to the nearest AP
def distance_to_nearest_AP(pos_of_user, list_of_AP):
    min = np.inf
    for x in list_of_AP:
        distance = np.sqrt((pos_of_user[0] - x[0]) * (pos_of_user[0] - x[0]) + (pos_of_user[1] - x[1]) * (pos_of_user[1] - x[1]))
        if distance < min:
            min = distance
    return min


#initialize user's postion with random value
#after initializing any user's position, check the distance from that user to the nearest AP,
#if the distance is satisfied, store it into the array list_of_users.
def initialize_users_pos():
    list_of_users = []
    i = 0
    while i < NUM_OF_USER:
        list_of_users.append((rd.uniform(0,length),rd.uniform(0,width)))
        if(distance_to_nearest_AP(pos_of_user= list_of_users[i], list_of_AP= list_of_AP) >= 0.001):
            i = i+1
        else:
            list_of_users.remove(list_of_users[i])
    return list_of_users

#The list contains positions of users
list_of_users=initialize_users_pos()

#Path loss model
def path_loss(distance):
    return 140.7 + 37.6*(np.log10(distance))

#Channel coefficient h=h_tilde* 10^(-pathloss/20)
#h_tilde = (a + b*i)/sqrt(2)
#in which a and b is random value from a Normal distribution
def generate_h_tilde(mu,sigma):
    re=np.random.normal(mu,sigma,1)[0]
    im=np.random.normal(mu,sigma,1)[0]
    h_tilde=complex(re,im)/np.sqrt(2)
    return h_tilde

def generate_h(user_index):
    h_tilde=generate_h_tilde(0,1)
    h=h_tilde* pow(10,-path_loss(distance_to_nearest_AP(list_of_users[user_index],list_of_AP))/20.0)#Nearest AP serves, will change later
    return h


#return a matrix of channel coefficient h between user k and AP b
def initialize_users_h():
    list_of_users_h=np.matrix(np.zeros([NUM_OF_AP,NUM_OF_USER]),dtype=complex)
    for b in range(NUM_OF_AP):
        for k in range (NUM_OF_USER):
            list_of_users_h[b,k]=generate_h(k)
    return list_of_users_h

# gamma_bkf (t) is the Signal to Interference-plus-Noise Ratio (SINR)
# at user k for the transmit signal from AP b, application f
def gamma(AP_index,user_index,application_index):
    h=initialize_users_h()
    power=pow(abs(h[AP_index,user_index]),2)*P
    interference_plus_noise=W*SIGMA_SQR
    for b in range(list_of_AP.__len__()):
        if(b!=AP_index):
            interference_plus_noise += pow(abs(h[b,user_index]),2)*P
    return power/interference_plus_noise

#achievable data rate r_bkf (t) for the link between
# AP b, user k and for application f using bandwidth Wf at scheduling frame t
def r(AP_index,user_index,application_index):
    return W*np.log2(1+gamma(AP_index,user_index,application_index))


#Plot APs and Users Position
plt.title("APs and Users Position")
AP_x,AP_y=zip(*list_of_AP)
User_x,User_y=zip(*list_of_users)
plt.scatter(AP_x,AP_y,cmap='hot')
plt.scatter(User_x,User_y,cmap='hot')
plt.grid()
plt.show(block=False)


#the value of r_bkf is immediate
#suppose in the real world, we have 10000 frames that users have to take action
#suppose the transmit power not depend on application f of user k -> r_bkf depends on which user k of AP b is?
#each frame has its r_bkf 
#Simulating 10000 frames, determine the value of r_bkf in each frame
NUM_OF_FRAME=100000
list_of_r_from_0_to_t = [[] for i in range(NUM_OF_FRAME)]
file = open("data_r.txt", "w")

for i in range(NUM_OF_FRAME):
    file.writelines(' FRAME START! '.center(200,'=')+"\n")
    for b in range(NUM_OF_AP):
        for k in range(NUM_OF_USER):
            r_bkf = r(b, k, application_index=1)
            file.write(f"{str(round(r_bkf,5)): <20}")
            list_of_r_from_0_to_t[i].append(r_bkf)
        file.writelines("\n")
    file.writelines(' END OF FRAME! '.center(200,'=')+"\n\n")

file.close()

plt.show()
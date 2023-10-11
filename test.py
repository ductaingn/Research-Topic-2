import numpy as np
import Enviroment as env
I=3
NUM_OF_DEVICE = 3
state_action = np.array([[0,1,2,3,4],
                [5,6,7,8,9]])

def generate_h_tidle(mu,sigma):
    h_tidle = []
    h_tidle_sub = np.empty(shape=(NUM_OF_DEVICE,env.NUM_OF_SUB_CHANNEL),dtype=complex)
    h_tidle_sub[:,:] = env.generate_h_tilde(mu,sigma)

    h_tidle_mW= np.empty(shape=(NUM_OF_DEVICE,env.NUM_OF_BEAM),dtype=complex)
    h_tidle_mW[:,:] = env.generate_h_tilde(mu,sigma)
    h_tidle.append(h_tidle_sub)
    h_tidle.append(h_tidle_mW)
    return h_tidle

print(generate_h_tidle(0,1)[0])
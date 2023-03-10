from env import *
import numpy as np
class KalmanFilter:
    def __init__(self, noise_velocity, noise_position) -> None:

       self.noise_position = noise_position
       self.noise_velocity = noise_velocity

       
       self.x_pred = np.random.randint(low=10,high=5000, size=(6,1))

       self.Q = self.noise_velocity * np.eye(6)
       self.R = self.noise_position * np.eye(3)
       self.A = np.eye(6)
       
       # self.A[0][3] = 1 = dt
       # self.A[1][4] = 1 = dt
       # self.A[2][5] = 1 =  dt
       # self.A[4][3] = 1 = dt
       # self.A[1][5] = 1 = dt
       
       self.A[0][3] = 1
       self.A[1][4] = 1
       self.A[2][5] = 1
       self.A[4][3] = 1
       self.A[1][5] = 1

       self.H = np.array([[1,0,0,0,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,1,0,0]])

       self.P = np.eye(6) #the covariance of the estimation error
       
       #                 np.array([[1,0,0],
       #                     [0,1,0],
       #                     [0,0,1],
       #                     [1,0,0],
       #                     [0,0,1],
       #                     [0,1,0]])
       
       self.B = np.array([[1,0,0],
                           [0,1,0],
                           [1,0,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0]])

    def input(self, observed_state:State, accel:numpy.ndarray, justUpdated:bool):
       accel = accel.reshape(3,1)

       if justUpdated:
              X = np.dot(self.P, self.H.T)
              Y = np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R)
              K_gain = np.dot(X,Y)
              # print("kgain shape: ",K_gain.shape)
              
              #incorporate new measurements
              self.z = np.array([observed_state.position[0],observed_state.position[1],observed_state.position[2]])
              self.z = self.z.reshape(3,1)

              self.res = self.z - np.dot(self.H, self.x_pred)
              
              self.x_pred = self.x_pred + np.dot(K_gain, self.res)
                 
       else: 
              #predict next state
              t1,t2 = np.dot(self.A, self.x_pred), np.dot(self.B, accel)
              self.x_pred = t1 + t2
              t3,t4 = np.dot(self.A, self.P), self.A.T
              self.P = np.dot(t3,t4) + self.Q

    def get_current_estimate(self)->State: # called just before new measurements come in
       #pred state x based on previous k measurements
       x_position = Vector3D(self.x_pred[0], self.x_pred[1], self.x_pred[2])
       x_velocity = Vector3D(self.x_pred[3], self.x_pred[4], self.x_pred[5])

       # print(x_position.shape)
       state_pred =  State(x_position.reshape(3), x_velocity.reshape(3))
       

       return state_pred
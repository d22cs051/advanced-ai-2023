import numpy as np

mat_transition = np.array([
    [1/3,1/3,0,0,1/3],
    [0,1/3,1/3,0,1/3],
    [0,0,0,1,0],
    [1,0,0,0,0],
    [1/3,1/3,0,0,1/3]
])


print(*mat_transition,sep="\n")
mat_transition1 = mat_transition.copy()
for i in range(10**6):
    mat_transition1 = np.matmul(mat_transition1,mat_transition)

print()
print("statonary probability distbution:") 
print(*mat_transition1,sep="\n")
print()
print("pi: ",mat_transition1[0])

print("-----------------------------")

pi = np.array([0,0,0,1,0])
new_pi = pi.copy()
for i in range(10**6):
    new_pi = np.matmul(new_pi,mat_transition)
print("-----------------------------")
print("pi:",*new_pi)





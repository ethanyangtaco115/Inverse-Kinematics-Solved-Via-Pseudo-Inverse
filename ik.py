import numpy as np

#link distances (placeholder for now) also in meters
L1 = 0.1
L2 = 0.1
L3 = 0.1

#defining rotation matrices, all in radians

def RotationZ(angle):
    c = np.np.cos(angle)
    s = np.np.sin(angle)
    return np.array([[c, -s, 0, 0],   
            [s,  c, 0, 0],  
            [0,  0, 1, 0],   
            [0,  0, 0, 1]])

def RotationY(angle):
    c = np.cos(angle)
    s = np.sin(angle) 
    return np.array([[ c, 0,  s, 0], 
            [ 0, 1,  0, 0],   
            [-s, 0,  c, 0], 
            [ 0, 0,  0, 1]])

def RotationX(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1,  0,  0, 0],
            [0,  c, -s, 0],
            [0,  s,  c, 0],
            [0,  0,  0, 1]])


def transfrom_joint_1(yaw_angle_1):
    c = np.cos(yaw_angle_1)
    s = np.sin(yaw_angle_1)

    T =  np.array([[c, -s, 0, 0],   
          [s,  c, 0, 0],
          [0,  0, 1, L1],   
          [0,  0, 0, 1]])
    
    return T

def transform_joint_2(roll_angle_2, pitch_angle_2):
    #roll around x axis
    cr = np.cos(roll_angle_2) #cr = cosine roll angle
    sr = np.sin(roll_angle_2) #sr = sine roll angle

    Roll_X = np.array([[1,  0,   0,  0],
              [0, cr, -sr, 0],
              [0, sr,  cr, 0],
              [0,  0,   0,  1]])
    
    #pitch around y axis
    cp = np.cos(pitch_angle_2) #haha cyberpunk abbreviated, also cp = cosine pitch angle
    sp = np.sin(pitch_angle_2) #sp = sine pitch angle

    Pitch_Y = np.array([[ cp, 0, sp, 0],
               [  0, 1,  0, 0],
               [-sp, 0, cp, 0],
               [  0, 0,  0, 1]])
    #translate along local z axis
    Translate = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, L2],
                 [0, 0, 0, 1]])
    
    #combine
    T2 = Roll_X @ Pitch_Y @ Translate

    return T2

def transform_joint_3(roll_angle_3, pitch_angle_3):
    #roll around x axis
    cr = np.cos(roll_angle_3) #cr = cosine roll angle
    sr = np.sin(roll_angle_3) #sr = sine roll angle

    Roll_X = np.array([[1,  0,   0,  0],
              [0, cr, -sr, 0],
              [0, sr,  cr, 0],
              [0,  0,   0,  1]])
    
    #pitch around y axis
    cp = np.cos(pitch_angle_3) #haha cyberpunk abbreviated, also cp = cosine pitch angle
    sp = np.sin(pitch_angle_3) #sp = sine pitch angle

    Pitch_Y = np.array([[ cp, 0, sp, 0],
               [  0, 1,  0, 0],
               [-sp, 0, cp, 0],
               [  0, 0,  0, 1]])
    #translate along local z axis
    Translate = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, L3],
                 [0, 0, 0, 1]])
    
    #combine
    T3 = Roll_X @ Pitch_Y @ Translate

    return T3

def fk(joint_angles):
    yaw1 = joint_angles[0]
    roll2 = joint_angles[1]
    pitch2 = joint_angles[2]
    roll3 = joint_angles[3]
    pitch3 = joint_angles[4]

    T1 = transfrom_joint_1(yaw1)
    T2 = transform_joint_2(roll2, pitch2)
    T3 = transform_joint_3(roll3, pitch3)

    T_total = T1 @ T2 @ T3

    end_position = np.array([T_total[0,3], T_total[1,3], T_total[2,3]])

    return end_position

def compute_jacobian(joint_angles):
    yaw1 = joint_angles[0]
    roll2 = joint_angles[1]
    pitch2 = joint_angles[2]
    roll3 = joint_angles[3]
    pitch3 = joint_angles[4]

    J = np.zeros((3,5))
    
    #build transformation matrices

    T1 = transfrom_joint_1(yaw1)
    T2 = T1 @ transform_joint_2(roll2, pitch2)
    T3 = T2 @ transform_joint_3(roll3, pitch3)

    #extract positions
    
    p0 = np.array([0, 0, 0])                   #base origin
    p1 = np.array([T1[0,3], T1[1,3], T1[2,3]]) # joint 1
    p2 = np.array([T2[0,3], T2[1,3], T2[2,3]]) # joint 2
    pe = np.array([T3[0,3], T3[1,3], T3[2,3]]) #end effector

    #extract joint axes

    z0 = np.array([0, 0, 1])                #base Z axis yaw
    x1 = np.array([T1[0,0], T1[1,0], T1[2,0]])        # joint 2 x axis (roll)
    y1 = np.array([T1[0,1], T1[1,1], T1[2,1]])        #joint 2 y axis (pitch)
    x2 = np.array([T2[0,0], T2[1,0], T2[2,0]])       #joint 3 x axis (roll)
    y2 = np.array([T2[0,1], T2[1,1], T2[2,1]])        #joint 3 y axis (pitch)

    #compute columns

    J[:, 0] = np.cross(z0, pe - p0)  # Column 0: Yaw1 effect
    J[:, 1] = np.cross(x1, pe - p1)  # Column 1: Roll2 effect
    J[:, 2] = np.cross(y1, pe - p1)  # Column 2: Pitch2 effect
    J[:, 3] = np.cross(x2, pe - p2)  # Column 3: Roll3 effect
    J[:, 4] = np.cross(y2, pe - p2)  # Column 4: Pitch3 effect

    return J

def solve_ik(joint_angles, desired_velocity): #psuedo inverse solving
    J = compute_jacobian(joint_angles) 

    J_pinv = np.linalg.pinv(J) #compute psuedo inverse oh wow numpy cheating

    joint_velocities = J_pinv @ desired_velocity

    return joint_velocities

def interate_ik(current_joint_angles, target_position, max_iterations=1000, tolerance=1):
    joint_angles = current_joint_angles.copy()
    step_size = 0.1
    dt = 0.01

    for i in range(max_iterations):
        current_position = fk(joint_angles) #compute current position
        position_error = target_position - current_position

        if np.linalg.norm(position_error) < tolerance:
            return joint_angles, True
        
        desired_velocity = step_size * position_error

        joint_velocities = solve_ik(joint_angles, desired_velocity)

        joint_angles += dt * joint_velocities

    return joint_angles, False

if __name__ == "__main__":
    # Test with some initial joint angles
    initial_angles = np.array([0.1, 0.2, 0.3, 0.1, 0.2])
    
    # Compute forward kinematics
    current_pos = fk(initial_angles)
    print(f"Current position: {current_pos}")
    
    # Test Jacobian computation
    J = compute_jacobian(initial_angles)
    print(f"Jacobian shape: {J.shape}")
    print(f"Jacobian:\n{J}")
    
    # Test pseudo-inverse IK
    target_pos = current_pos + np.array([10, 10, 0])  # Move 10mm in each direction
    solution, success = interate_ik(initial_angles, target_pos)
    
    if success:
        print(f"IK solution found: {solution}")
        final_pos = fk(solution)
        print(f"Final position: {final_pos}")
        print(f"Position error: {np.linalg.norm(target_pos - final_pos)}")
    else:
        print("IK solution not found within tolerance")
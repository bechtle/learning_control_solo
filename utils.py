import numpy as np
import pybullet as p
import pinocchio as se3


def get_euler(q):
    quaternion = q[3:3+4]
    angle = np.array(p.getEulerFromQuaternion(quaternion))
    new_q = np.hstack([q[:3],angle,q[7:]])
    return new_q


def convert_local(global_states):
    local_states = []
    for state in global_states:
        state = list(state)
        q = state[:12+7]
        dq = state[12+7:]
        rot = np.array(p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
        dq[0:3] = rot.dot(dq[0:3])
        dq[3:6] = rot.dot(dq[3:6])
        local_state = np.hstack([q, dq])
        local_states.append(local_state)
    return np.array(local_states)


# def process_state(dq,ddq,forces,dt,rot):
#
#     # Pinocchio assumes the base velocity to be in the body frame -> we want them in world frame.
#     ddq[0:3] = rot.dot(ddq[0:3])
#     ddq[3:6] = rot.dot(ddq[3:6])
#     dq = dq + ddq * dt
#
#     dq[:3] = dq[:3] * 10
#     dq = np.delete(dq, [0, 1, 3, 4, 5])
#     dq *= 10
#
#     return np.hstack([dq,forces])
#
#
#
# def process_robot_state(q,dq,forces,robot):
#     # Pinocchio assumes the base velocity to be in the body frame -> we want them in world frame.
#     rot = np.array(p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
#     dq[0:3] = rot.dot(dq[0:3])
#     dq[3:6] = rot.dot(dq[3:6])
#
#     q = get_euler(q)
#     q[:3] = q[:3] * 10
#     q = np.delete(q, [0, 1, 3, 4, 5])
#     dq = np.delete(dq, [0, 1, 3, 4, 5])
#     q *= 10
#
#     ee_index, measured_forces = forces
#     forces = np.zeros([len(robot.pinocchio_endeff_ids), 6])
#     binary_forces = np.zeros([len(robot.pinocchio_endeff_ids)])
#     for idx, active_eff in enumerate(ee_index):
#         array_index = robot.pinocchio_endeff_ids.index(active_eff)
#         binary_forces[array_index] = 1.0
#         forces[array_index] = np.array(forces)[idx] / 100.0
#     forces = forces.reshape([4, 3])
#     return np.hstack([q,dq,forces.squeeze()])


def process_robot_state(q,dq,forces,robot):
    h = se3.rnea(robot.pin_robot.model, robot.pin_robot.data, q, dq, np.zeros_like(dq))
    M = se3.crba(robot.pin_robot.model, robot.pin_robot.data, q)

    # Pinocchio assumes the base velocity to be in the body frame -> we want them in world frame.
    rot = np.array(p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
    dq[0:3] = rot.dot(dq[0:3])
    dq[3:6] = rot.dot(dq[3:6])

    q = get_euler(q)

    ee_index, measured_forces = forces
    forces = np.zeros([len(robot.pinocchio_endeff_ids), 3])
    binary_forces = np.zeros([len(robot.pinocchio_endeff_ids)])
    for idx, active_eff in enumerate(ee_index):
        array_index = robot.pinocchio_endeff_ids.index(active_eff)
        binary_forces[array_index] = 1.0
        forces[array_index] = np.array(measured_forces)[idx,:3]/100.0
    forces = forces.reshape([4*3])
    return q,dq,forces,M,h,rot
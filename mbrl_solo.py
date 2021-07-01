import numpy as np
from model_module import ForwardModel, InverseModel

import pybullet as p
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot
from robot_properties_solo.config import Solo12Config

from utils import process_robot_state
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
global log_iter
import time
log_iter = 0
import sys
import dill as pickle
import uuid

def init_models(robot,f_model,i_model,des_dq,des_forces,des_q):
    global log_iter
    #rand_taus = np.random.uniform(low=-torque_limits/5.0, high=torque_limits/5.0, size=(150, 12))
    taus = np.load('demos/des_taus_200.npy').squeeze()[:50]

    q = np.array(Solo12Config.initial_configuration)
    dq = np.array(Solo12Config.initial_velocity)
    robot.reset_state(q.T, dq.T)
    p.stepSimulation()



    q, dq = robot.get_state()
    forces = robot.get_force()

    _q = []
    _dq = []
    _forces = []
    _M = []
    _h = []
    _rot = []
    _taus = []

    q,dq,forces,M,h,rot = process_robot_state(q, dq, forces,robot)


    _q.append(q)
    _dq.append(dq)
    _forces.append(forces)
    _M.append(M)
    _h.append(h)
    _rot.append(rot)

    for t,tau in enumerate(taus):
        tau = ((3.0 * (des_q[0][6:] - q[6:]) + (0.1 * (des_dq[0][6:] - dq[6:])))).clip(-1.0, 1.0)
        robot.send_joint_command(tau)
        # Step the simulator.
        p.stepSimulation()

        # Read the final state and forces after the stepping.
        q, dq = robot.get_state()
        forces = robot.get_force()

        q,dq,forces,M,h,rot = process_robot_state(q, dq, forces,robot)
        _q.append(q)
        _dq.append(dq)
        _forces.append(forces)
        _M.append(M)
        _h.append(h)
        _rot.append(rot)
        _taus.append(tau)
    #
    # np.save('demos/des_dq_200',_dq)
    # np.save('demos/des_forces_200',_forces[1:])
    # np.save('demos/des_taus_200',taus)
    # np.save('demos/des_q_200',_q[1:])
    # exit()

    f_model.train(_q,_dq,_M,_h,_rot,_forces,_taus)
    i_model.train_coupled(_q,_dq,_forces,_taus,_M,_h,_rot,des_dq=des_dq,des_forces=des_forces,f_model=f_model,joint_loss=True,sup_loss=False)

    # logging:
    # pred_dq,pred_forces = f_model.forward(torch.Tensor(_q)[:-1],torch.Tensor(_dq)[:-1],torch.Tensor(_M)[:-1],torch.Tensor(_h)[:-1],torch.Tensor(_rot)[:-1],torch.Tensor(_forces)[:-1],torch.Tensor(_taus))
    #
    # des_states = f_model._process_state(torch.Tensor(_dq)[1:])*f_model.dt
    #
    # for n_iter in range(len(pred_dq)):
    #     for dim in range(len(pred_dq[n_iter])):
    #         writer.add_scalars('dim '+str(dim) , {'pred': pred_dq[n_iter, dim],
    #                                        'obs': des_states[n_iter, dim]},log_iter
    #                            )
    #     for dim in range(len(pred_forces[n_iter])):
    #         writer.add_scalars('forces '+str(dim) , {'pred': pred_forces[n_iter, dim],
    #                                        'obs': torch.Tensor(_forces)[n_iter, dim]},log_iter
    #                            )
    #     log_iter += 1
    return None


if __name__ == "__main__":

    experiment_id = uuid.uuid4()

    np.random.seed(422)
    torch.manual_seed(422)

    dt = 1.0/250.0
    env = BulletEnvWithGround(p.GUI,dt=dt)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, cameraPitch=-30,
                                 cameraTargetPosition=[0.5, 0, 0])
    robot = env.add_robot(Solo12Robot)
    dt = 1.0/250.0

    coupled = True
    target_index = '200'
    joint_loss = True

    torque_limits = np.zeros(12)+1.0
    des_dq = list(np.load('demos/des_dq_'+str(target_index)+'.npy'))
    des_forces = list(np.load('demos/des_forces_'+str(target_index)+'.npy'))
    des_q = list(np.load('demos/des_q_' + str(target_index) + '.npy'))
    base_dim = 6
    f_model = ForwardModel(torque_limits=torque_limits,base_dim=base_dim,force_dim=12,robot=robot,dt=dt,dof=12)
    i_model = InverseModel(torque_limits=torque_limits,base_dim=base_dim,force_dim=12,robot=robot,dt=dt,dof=12)


    reset_state = init_models(robot,f_model,i_model,des_dq,des_forces,des_q)


    mbrl_iterations =30

    res_dic = {}
    res_dic['target'] = target_index
    res_dic['coupled'] = coupled
    res_dic['joint_loss'] = joint_loss
    res_dic['mbrl_iterations'] = []

    for mbrl_iter in range(mbrl_iterations):
        iter_dic = {}

        q = np.array(Solo12Config.initial_configuration)
        dq = np.array(Solo12Config.initial_velocity)
        robot.reset_state(q.T, dq.T)
        p.stepSimulation()

        q, dq = robot.get_state()
        forces = robot.get_force()

        _q = []
        _dq = []
        _forces = []
        _M = []
        _h = []
        _rot = []
        _taus = []

        q, dq, forces, M, h, rot = process_robot_state(q, dq, forces, robot)

        _q.append(q)
        _dq.append(dq)
        _forces.append(forces)
        _M.append(M)
        _h.append(h)
        _rot.append(rot)

        feedback_state = q.copy()


        for t in range(len(des_dq)-1):
            pred_tau, PD = i_model.predict(q,dq,des_dq[t+1],forces,des_forces[t],des_q[0])
            robot.send_joint_command(pred_tau)
            # Step the simulator.
            p.stepSimulation()
            feedback_state = q.copy()
            # Read the final state and forces after the stepping.
            q, dq = robot.get_state()
            forces = robot.get_force()

            q, dq, forces, M, h, rot = process_robot_state(q, dq, forces, robot)
            _q.append(q)
            _dq.append(dq)
            _forces.append(forces)
            _M.append(M)
            _h.append(h)
            _rot.append(rot)
            _taus.append(pred_tau)


        # logging:
        pred_dq, pred_forces,pred_ddq = f_model.forward(torch.Tensor(_q)[:-1], torch.Tensor(_dq)[:-1], torch.Tensor(_M)[:-1],
                                               torch.Tensor(_h)[:-1], torch.Tensor(_rot)[:-1],
                                               torch.Tensor(_forces)[:-1], torch.Tensor(_taus))

        obs_states = f_model._process_state(torch.Tensor(_dq)[1:]) * f_model.dt
        norm_des_dq = f_model._process_state(torch.Tensor(des_dq))*f_model.dt

        for n_iter in range(len(pred_dq)):
            for dim in range(len(pred_dq[n_iter])):
                writer.add_scalars('dim ' + str(dim), {'pred': pred_dq[n_iter, dim],
                                                       'obs': obs_states[n_iter, dim],
                                                       'des': norm_des_dq[n_iter,dim]}, log_iter
                                   )
            for dim in range(len(pred_forces[n_iter])):
                writer.add_scalars('forces ' + str(dim), {'pred': pred_forces[n_iter, dim],
                                                          'obs': torch.Tensor(_forces)[n_iter, dim],
                                                          'des':torch.Tensor(des_forces)[n_iter,dim]}, log_iter
                                   )
            log_iter += 1

        f_model.train(_q, _dq, _M, _h, _rot, _forces, _taus)
        i_model.train_coupled(_q, _dq, _forces, _taus, _M, _h, _rot, des_dq=des_dq, des_forces=des_forces,
                              f_model=f_model, joint_loss=True, sup_loss=False)

       # ####logging:
       #
       #  pred_delta = pred.detach().numpy()
       #
       #  for n_iter in range(len(pred_delta)):
       #      for dim in range(len(pred_delta[n_iter])):
       #          writer.add_scalars('dim ' + str(dim), {'pred': pred_delta[n_iter, dim],
       #                                                 'obs': obs_des_states[n_iter, dim],
       #                                                 'des': q_des[n_iter,dim]}, log_iter)
       #      for dim in range(12):
       #          writer.add_scalars('tau '+str(dim), {'taus': taus[n_iter,dim]},log_iter)
       #
       #
       #      for dim in range(f_model.pos_dim):
       #          writer.add_scalars('pos ' + str(dim), {'obs': global_states[n_iter, dim],
       #                                                 'des': des_global_state[n_iter,dim]}, log_iter)
       #      log_iter+=1
       #
       #  mse_loss = torch.nn.MSELoss()
       #
       #  pred_error = mse_loss(pred,torch.Tensor(obs_des_states))
       #  perf_error = mse_loss(torch.Tensor(obs_des_states),torch.Tensor(q_des))
       #  task_error = mse_loss(torch.Tensor(q_des),pred)
       #  base_error = np.mean((global_states[:,0]-des_global_state[:,0])**2)
       #
       #  writer.add_scalar('base error',base_error,mbrl_iter)
       #  writer.add_scalar('pred error',pred_error.item()*100,mbrl_iter)
       #  writer.add_scalar('perf error', perf_error.item(), mbrl_iter)
       #  writer.add_scalar('task error', task_error.item()*100, mbrl_iter)
       #
       #
       #  iter_dic['it_count'] = mbrl_iter
       #  iter_dic['states'] = global_states
       #  iter_dic['obs_accs'] = obs_des_states
       #  iter_dic['actions'] = taus
       #  iter_dic['pred_states'] = pred_delta
       #
       #  iter_dic['base_error'] = base_error
       #  iter_dic['pred_error'] = pred_error.item()
       #  iter_dic['perf_error'] = perf_error.item()
       #  iter_dic['task_error'] = task_error.item()
       #
       #
       #  iter_dic['q_des'] = np.array(q_des)
       #  iter_dic['des_states'] = des_global_state
       #
       #  iter_dic['pred_losses'] = f_model.pred_losses.copy()
       #  iter_dic['task_losses'] = f_model.task_losses.copy()
       #  iter_dic['perf_losses'] = f_model.perf_losses.copy()
       #  iter_dic['f_model'] = f_model.save().copy()
       #  iter_dic['i_model'] = i_model.save().copy()
       #  del f_model.pred_losses
       #  del f_model.task_losses
       #  del f_model.perf_losses
       #  f_model.pred_losses = []
       #  f_model.task_losses =[]
       #  f_model.perf_losses = []
       #
       #  res_dic['mbrl_iterations'].append(iter_dic)

        # with open('results/jordan/res_dict_'+str(experiment_id)+'.pkl', "wb") as fp:
        #     pickle.dump(res_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)


        # X_data = f_model.X
        # Y_data = f_model.Y
        # f_model = ForwardModel(pos_dim=1 + 12 + (4 * 3), vel_dim=1 + 12, action_dim=12, output_dim=1 + 12 + (4 * 3),
        #                        torque_limits=torque_limits, babbling_length=babbling_length)
        # f_model.X = X_data.copy()
        # f_model.Y = Y_data.copy()



# Expose the name also as Solo12Robot.
#Solo12Robot = Quadruped12Robot

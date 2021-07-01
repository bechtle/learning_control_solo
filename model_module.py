import numpy as np
import time
import torch
from termcolor import colored
import torch.nn as nn

class Dataset_InvModel(torch.utils.data.Dataset):
    def __init__(self, q,dq,M,h,rot,forces,tau,y_dq,y_forces,next_qd,next_forces):
        self.dataset = [
            (torch.Tensor(q[i]), torch.Tensor(dq[i]),torch.Tensor(M[i]),torch.Tensor(h[i]),torch.Tensor(rot[i]),torch.Tensor(forces[i]),torch.Tensor(tau[i]),torch.Tensor(y_dq[i]),torch.Tensor(y_forces[i]),torch.Tensor(next_qd[i]),torch.Tensor(next_forces[i])) for i in range(len(q))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class InverseModel(torch.nn.Module):
    def __init__(self, torque_limits, base_dim, force_dim, robot,dt, dof = 12):
        super(InverseModel, self).__init__()

        input_dim = base_dim+dof+base_dim+dof+force_dim+base_dim+dof+force_dim
        self.output_dim = dof
        self.robot = robot
        self.torque_limits = torque_limits
        self.base_dim = base_dim


        self.learning_rate = 0.001
        self.display_epoch = 50
        self.epochs = 400

        self.norm_in = np.ones(input_dim)
        self.norm_in[base_dim+dof:(base_dim+dof)*2]=8.0
        self.norm_in[(base_dim+dof)*2+force_dim:(base_dim+dof)*2+force_dim+dof] = 8.0
        self.norm_in = np.expand_dims(self.norm_in,axis=0)
        self.norm_in = torch.Tensor(self.norm_in)
        self.dt = dt

        self.training_set = {}
        self.training_set['q'] = []
        self.training_set['dq'] = []
        self.training_set['next_dq'] = []
        self.training_set['M'] = []
        self.training_set['h'] = []
        self.training_set['rot'] = []
        self.training_set['tau'] = []
        self.training_set['forces'] = []
        self.training_set['next_forces'] = []
        self.training_set['des_dq'] = []
        self.training_set['des_forces'] = []



        w = [300,300,300]
        activation = nn.Softsign
        self.layers = nn.Sequential(
            nn.Linear(input_dim, w[0]),
            activation(),
            nn.BatchNorm1d(num_features=w[0]),
            nn.Linear(w[0], w[1]),
            activation(),
            nn.BatchNorm1d(num_features=w[1]),
            nn.Linear(w[1], w[2]),
            activation(),
            nn.BatchNorm1d(num_features=w[2]),
            nn.Linear(w[2], self.output_dim)
        )

        self.Kd = torch.Tensor(0.0 * torch.ones(12))
        self.Kp = torch.Tensor(0.0 * torch.ones(12)) #3.0

    def _process_state(self,joints):
        #joints[:,:3] = joints[:,:3] * 10
        #joints = torch.cat((joints[:,2].unsqueeze(dim=1),joints[:,6:]),dim=1)
        #joints*=10
        return joints

    def _normalize(self,q,dq,forces,des_qd,des_forces):
        inputx = torch.cat((self._process_state(q),self._process_state(dq),forces,self._process_state(des_qd),des_forces),dim=1)/self.norm_in
        return inputx

    def forward(self,q,dq,des_dq,forces,des_forces):
        inputx = self._normalize(q,dq,forces,des_dq,des_forces)
        pred_tau = self.layers(inputx)
        return pred_tau.clamp(-1.0,1.0)

    def train_coupled(self,q,dq,forces,tau,M,h,rot,des_dq,des_forces,f_model,joint_loss=False,sup_loss=False):
        self.train()
        self.training_set['q'] += q[:-1]
        self.training_set['dq'] += dq[:-1]
        self.training_set['next_dq'] +=dq[1:]
        self.training_set['M'] += M[:-1]
        self.training_set['h'] += h[:-1]
        self.training_set['rot'] += rot[:-1]
        self.training_set['tau'] += tau
        self.training_set['forces'] += forces[:-1]
        self.training_set['next_forces'] += forces[1:]
        self.training_set['des_dq'] += des_dq[1:]
        #self.training_set['des_dq'] += list((np.array(des_dq[1:]) - np.array(des_dq[:-1]))/self.dt)
        self.training_set['des_forces'] += des_forces



        training_data = Dataset_InvModel(self.training_set['q'],self.training_set['dq'],self.training_set['M'],self.training_set['h'],self.training_set['rot'],
                                self.training_set['forces'],self.training_set['tau'],self.training_set['des_dq'],self.training_set['des_forces'],self.training_set['next_dq'],self.training_set['next_forces'])

        train_loader = torch.utils.data.DataLoader(
            training_data, batch_size=10000, num_workers=0, shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        mse_loss = torch.nn.MSELoss()

        for epoch in range(self.epochs):
            losses = []
            for batch, (q,dq,M,h,rot,forces,taus,des_dq,des_forces,next_dq,next_forces) in enumerate(train_loader, 1):
                pred_tau = self.forward(q,dq,next_dq,forces,next_forces)
                pred_qd, pred_forces,pred_ddq = f_model.forward(q,dq,M,h,rot,forces,pred_tau)

                pred_loss  = torch.mean(torch.norm(pred_qd-self._process_state(next_dq)*self.dt,dim=1))+torch.mean(torch.norm(pred_forces-next_forces,dim=1)) +\
                             10*torch.mean(torch.norm((pred_qd[:,2] - self._process_state(next_dq)[:,2]*self.dt), dim=0))
                force_loss = torch.mean(torch.norm(pred_forces - des_forces, dim=1))
                task_loss = torch.mean(torch.norm((pred_qd - self._process_state(des_dq)*self.dt), dim=1)) + force_loss + 10*torch.mean(torch.norm((pred_qd[:,2:6] - self._process_state(des_dq)[:,2:6]*self.dt), dim=1))

                #task_loss = 10*torch.mean(torch.norm((pred_qd[:,2] - self._process_state(des_dq)[:,2]*self.dt), dim=0)) + torch.mean(torch.norm((pred_qd[:,:6] - self._process_state(des_dq)[:,:6]*self.dt), dim=1))

                # pred_loss = mse_loss(pred_qd,self._process_state(next_dq)*self.dt) + mse_loss(pred_forces,next_forces)
                # force_loss = mse_loss(pred_forces,des_forces)
                # task_loss = mse_loss(pred_qd,self._process_state(des_dq)*self.dt) + force_loss

                loss = task_loss + pred_loss
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if epoch % self.display_epoch == 0:
                print(
                    colored(
                        "epoch={}, loss={}, task={}, pred={}".format(epoch, np.mean(losses),task_loss.item(),pred_loss.item()), "yellow"
                    )
                )


    def _augment_state(self,joint):
        joint = torch.Tensor(joint).unsqueeze(dim=0)
        return joint

    def predict(self,q,dq,des_dq,forces,des_forces,feedback_term):
        self.eval()
        pred_tau = self.forward(self._augment_state(q),self._augment_state(dq),self._augment_state(des_dq),self._augment_state(forces),self._augment_state(des_forces))
        pred_tau = pred_tau.clamp(-1.0,1.0).squeeze()
        PD = torch.zeros_like(pred_tau)
        for i in range(len(PD)):
            aug_feedback = self._process_state(self._augment_state(feedback_term)).squeeze()
            aug_q = self._process_state(self._augment_state(q)).squeeze()
            aug_qd =  self._process_state(self._augment_state(dq)).squeeze()
            PD[i] = self.Kp[i]*(aug_feedback[i+self.base_dim]-aug_q[i+self.base_dim]) - self.Kd[i]*(aug_qd[i+self.base_dim]*self.dt)
            PD = PD.clamp(-1.0,1.0)

        return (pred_tau+PD).detach().numpy(),PD.detach().numpy()

    def save(self):
        return self.ensemble.save_model()



class Dataset(torch.utils.data.Dataset):
    def __init__(self, q,dq,M,h,rot,forces,tau,y_dq,y_forces):
        self.dataset = [
            (torch.Tensor(q[i]), torch.Tensor(dq[i]),torch.Tensor(M[i]),torch.Tensor(h[i]),torch.Tensor(rot[i]),torch.Tensor(forces[i]),torch.Tensor(tau[i]),torch.Tensor(y_dq[i]),torch.Tensor(y_forces[i])) for i in range(len(q))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class ForwardModel(torch.nn.Module):

    def __init__(self,torque_limits,base_dim,force_dim,robot,dt,dof=12):
        super(ForwardModel, self).__init__()

        input_dim = base_dim+dof+base_dim+dof+force_dim+dof
        self.pos_dim = base_dim+dof
        self.force_dim = force_dim
        self.state_dim = (base_dim+dof)*2
        self.robot = robot
        self.learning_rate = 0.001
        self.display_epoch=50
        self.epochs = 500

        self.norm_in = np.ones(input_dim)
        self.torque_limits = np.array(torque_limits)
        self.norm_in[self.pos_dim:self.state_dim] = 8.0
        self.norm_in[self.state_dim+self.force_dim:] = torque_limits.copy()
        self.norm_in = np.expand_dims(self.norm_in,axis=0)
        self.norm_in = torch.Tensor(self.norm_in)
        self.dt = dt


        self.training_set = {}
        self.training_set['q'] = []
        self.training_set['dq'] = []
        self.training_set['M'] = []
        self.training_set['h'] = []
        self.training_set['tau'] = []
        self.training_set['forces'] = []
        self.training_set['target_dq'] = []
        self.training_set['target_forces'] = []
        self.training_set['rot'] = []

        w = [1000, 500,500]
        activation = nn.ReLU

        self.lag_froces = nn.Sequential(
            nn.Linear(input_dim, w[0]),
            activation(),
            nn.Linear(w[0], w[1]),
            activation(),
            nn.Linear(w[1], w[2]),
            activation(),
            nn.Linear(w[2], 18),
        )

        self.pred_forces = nn.Sequential(
            nn.Linear(input_dim, w[0]),
            activation(),
            nn.Linear(w[0], w[1]),
            activation(),
            nn.Linear(w[1], w[2]),
            activation(),
            nn.Linear(w[2], 12),
        )

    def _process_state(self,joints):
        #joints[:,:3] = joints[:,:3] * 10
        #joints = torch.cat((joints[:,2].unsqueeze(dim=1),joints[:,6:]),dim=1)
        #joints*=10
        return joints

    def _normalize(self,q,dq,forces,tau):
        inputx = torch.cat((self._process_state(q),self._process_state(dq),forces,tau),dim=1)/self.norm_in
        return inputx

    def forward(self,q,dq,M,h,rot,forces,tau):
        input = self._normalize(q,dq,forces,tau)
        forces_term = self.lag_froces(input)

        filled_tau = torch.cat((torch.zeros_like(tau)[:,:6],tau),dim=1)
        b = (filled_tau-h).unsqueeze(dim=2)
        ddq = torch.solve(b,M)[0].squeeze() - forces_term
        ddq = ddq.unsqueeze(dim=2)
        # rotate to world frame
        ddq[:,0:3] = rot.bmm(ddq[:,0:3])
        ddq[:,3:6] = rot.bmm(ddq[:,3:6])

        ddq = ddq.squeeze(-1)*self.dt

        dq = dq + ddq
        dq = self._process_state(dq)

        forces = self.pred_forces(input)
        return dq*self.dt,forces,ddq


    def train(self,q,dq,M,h,rot,forces,tau):

        self.training_set['q'] += q[:-1]
        self.training_set['dq'] += dq[:-1]
        self.training_set['M'] += M[:-1]
        self.training_set['h'] += h[:-1]
        self.training_set['rot'] += rot[:-1]
        self.training_set['tau'] += tau
        self.training_set['forces'] += forces[:-1]
        self.training_set['target_dq'] += list((np.array(dq[1:]) - np.array(dq[:-1]))/self.dt)
        self.training_set['target_forces'] += forces[1:]



        training_data = Dataset(self.training_set['q'],self.training_set['dq'],self.training_set['M'],self.training_set['h'],self.training_set['rot'],self.training_set['forces'],self.training_set['tau'],self.training_set['target_dq'],self.training_set['target_forces'])

        train_loader = torch.utils.data.DataLoader(
            training_data, batch_size=1000, num_workers=0, shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        mse_loss = torch.nn.MSELoss()

        for epoch in range(self.epochs):
            losses = []
            for batch, (q,dq,M,h,rot,obs_force,tau,target_dq,target_forces) in enumerate(train_loader, 1):

                pred_dq,pred_force,pred_ddq = self.forward(q,dq,M,h,rot,obs_force,tau)

                target_dq = self._process_state(target_dq)*self.dt
                loss_dq = mse_loss(pred_ddq,target_dq)
                loss_forces = mse_loss(pred_force,target_forces)

                loss = loss_dq+loss_forces
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if epoch % self.display_epoch == 0:
                print(
                    colored(
                        "epoch={}, loss={}".format(epoch, np.mean(losses)), "yellow"
                    )
                )


    def get_gradients(self,state,action):
        X = np.append(state,action)/self.norm_in
        return self.ensemble.get_gradient(X)

    def save(self):
        return self.ensemble.save_model()

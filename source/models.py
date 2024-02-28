import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from source.utils import wrap_to_pi, rk4_step_func_autonomous
import time
import math

def model_factory(model_class, kwargs) -> "DynamicsModel":
    model_class = globals()[model_class]
    model = model_class(**kwargs)
    return model


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        return self.linear3(h)

class MLP_acr(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        h = F.relu(self.linear3(h))
        return self.linear4(h)

class DynamicsModel_ctcartpole(nn.Module):
    def __init__(self, integration_scheme="symplectic", dt=0.02, wrap_angle=False, **kwargs):
        super().__init__()
        self.integration_scheme = integration_scheme
        self.dt = dt
        self.wrap_angle = wrap_angle

    def derivative(self, state, action):
        raise NotImplementedError

    def get_Fa(self):
        raise NotImplementedError

    @staticmethod
    def wrap_state(state):
        wrapped = state.clone()
        wrapped[..., 2] = wrap_to_pi(state[..., 2])
        return wrapped

    def forward(self, state, action):


        w_state = self.wrap_state(state) if self.wrap_angle else state

        if self.integration_scheme == "euler":
            return state + self.derivative(w_state, action) * self.dt
        elif self.integration_scheme == "rk4":
            func = partial(self.derivative, action=action)
            derivative = rk4_step_func_autonomous(func=func, dt=self.dt, y=w_state)
            new_state = state + derivative * self.dt
            return self.wrap_state(new_state)
        elif self.integration_scheme == "symplectic":
            assert state.shape[1] % 2 == 0  
            p = state.shape[1] // 2
            new_state = state + self.derivative(w_state, action) * self.dt  # Euler
            new_state[:, 0] = state[:, 0] + new_state[:, 1] * self.dt 
            new_state[:, 2] = state[:, 2] + new_state[:, 3] * self.dt
            return new_state

    def integrate(self, state, actions, resets=None):
        """
        Integrate a trajectory
        :param state: initial state, of shape batch x state size
        :param actions: sequence of actions, of shape: horizon x batch x action size
        :param resets: dict of (timestep, state) such that the trajectory at time timestep is reset to state
        :return: resulting trajectory, of shape: horizon x batch x state size
        """
        states = torch.zeros((actions.shape[0], state.shape[0], state.shape[1])).to(state.device)
        with torch.no_grad():
            for t in range(actions.shape[0]):
                state = self.forward(state, actions[t, ...])
                if resets and t in resets:
                    state = resets[t]
                states[t, ...] = state
        return states




class MLPModel_ctcartpole(DynamicsModel_ctcartpole):
    def __init__(self, state_size, action_size, hidden_size=16, **kwargs):
        super().__init__(**kwargs)
        self.state_size, self.action_size = state_size, action_size
        self.mlp = MLP(state_size + action_size, hidden_size, state_size)

    def derivative(self, state, action):
        """
            Predict dx_t = MLP(x_t,u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """

        xu = torch.cat((state, action), -1)
        return self.mlp(xu)

class PerfectCartPole(DynamicsModel_ctcartpole):
    def __init__(self, device="cuda", with_friction=True, integration_scheme="symplectic", **kwargs):
        super().__init__(integration_scheme=integration_scheme, **kwargs)
        self.device = device
        self.force_max = 10.0
        self.g = 9.8
        self.m1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float).to(self.device), requires_grad=False)
        self.m2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float).to(self.device), requires_grad=False)
        self.l = nn.Parameter(torch.tensor(0.5, dtype=torch.float).to(self.device), requires_grad=False)

        
        if with_friction:
            self.k1 = nn.Parameter(torch.tensor(0.1, dtype=torch.float).to(self.device), requires_grad=False)
            self.k2 = nn.Parameter(torch.tensor(0.02, dtype=torch.float).to(self.device), requires_grad=False)
        else:
            self.k1 = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)
            self.k2 = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)

    def derivative(self, state, action): 


        dx = torch.zeros(state.shape).to(state.device)
        dx[:, 0] = state[:, 1]
        dx[:, 2] = state[:, 3]
        force = action[:, 0] * self.force_max
        costheta = torch.cos(state[:, 2])
        sintheta = torch.sin(state[:, 2])

        temp = (
            force + (self.l * self.m2 * state[:, 3]**2 * sintheta) - (self.k1 * torch.sign(state[:, 1]))
            ) / (self.m1 + self.m2)

        dx[:, 3] = ((self.g * sintheta - costheta * temp) - ((self.k2 * state[:, 3]) /(self.l * self.m2))) / (self.l * (4.0 / 3.0 - self.m2 * costheta**2 / (self.m1 + self.m2)))
        dx[:, 1] = temp - (self.l * self.m2 * ((self.g * sintheta - costheta * temp) - ((self.k2 * state[:, 3]) /(self.l * self.m2))) / (self.l * (4.0 / 3.0 - self.m2 * costheta**2 / (self.m1 + self.m2))) * costheta) / (self.m1 + self.m2)


        return dx     
     
    def get_params(self):

        return {"Cart_mass 1.0": self.m1.data,
                "Pole_mass 0.1": self.m2.data,
                "Longueur 0.5": self.l.data,
                "friction1 0.1":self.k1.data,
                "friction2 0.02":self.k2.data}      
        
class CartPole(DynamicsModel_ctcartpole):
    def __init__(self, device="cuda", with_friction=True, integration_scheme="symplectic", **kwargs):
        super().__init__(integration_scheme=integration_scheme, **kwargs)
        self.device = device
        self.force_max = 10.0
        self.g = 9.8
        self.m1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float).to(self.device), requires_grad=True)
        self.m2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float).to(self.device), requires_grad=True)
        self.l = nn.Parameter(torch.tensor(0.5, dtype=torch.float).to(self.device), requires_grad=True)

        
        if with_friction:
            self.k1 = nn.Parameter(torch.tensor(0.1, dtype=torch.float).to(self.device), requires_grad=True)
            self.k2 = nn.Parameter(torch.tensor(0.02, dtype=torch.float).to(self.device), requires_grad=True)
        else:
            self.k1 = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)
            self.k2 = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)

    def derivative(self, state, action): 

        
        


        dx = torch.zeros(state.shape).to(state.device)
        dx[:, 0] = state[:, 1]
        dx[:, 2] = state[:, 3]
        force = action[:, 0] * self.force_max
        costheta = torch.cos(state[:, 2])
        sintheta = torch.sin(state[:, 2])



        temp = (
            force + (self.l * self.m2 * state[:, 3]**2 * sintheta) - (self.k1 * torch.sign(state[:, 1]))
            ) / (self.m1 + self.m2)


        dx[:, 3] = ((self.g * sintheta - costheta * temp) - ((self.k2 * state[:, 3]) /(self.l * self.m2))) /(self.l * (4.0 / 3.0 - self.m2 * costheta**2 / (self.m1 + self.m2)))

        dx[:, 1] = temp - (self.l * self.m2 * ((self.g * sintheta - costheta * temp) - ((self.k2 * state[:, 3]) /(self.l * self.m2))) /(self.l * (4.0 / 3.0 - self.m2 * costheta**2 / (self.m1 + self.m2))) * costheta) / (self.m1 + self.m2)

        return dx     
     
    def get_params(self):

        return {"Cart_mass 1.0": self.m1.data,
                "Pole_mass 0.1": self.m2.data,
                "Longueur 0.5": self.l.data,
                "friction1 0.1":self.k1.data,
                "friction2 0.02":self.k2.data}        

class FrictionlessCartPole(CartPole):
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        kwargs["with_friction"] = False
        super().__init__(**kwargs)


class AugmentedCartPole(DynamicsModel_ctcartpole):
    """
    Frictionless pendulum
    """
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        super().__init__(integration_scheme=integration_scheme, **kwargs)
        self.device = device
        self.cartpole = FrictionlessCartPole(**kwargs)
        self.mlp = MLPModel_ctcartpole(**kwargs)
        self.Fa = torch.tensor(0)
        self.Fp = torch.tensor(0)

    def derivative(self, state, action):
        self.Fp = self.cartpole.derivative(state, action)
        self.Fa = self.mlp.derivative(state, action)  
        return self.Fp + self.Fa

    def get_Fa(self):
        return self.Fa, self.Fp


class Aphynity_ctcartpole(AugmentedCartPole):
    """
    Frictionless pendulum
    """
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        super().__init__(**kwargs)
        self.use_aphynity = True


class DynamicsModel(nn.Module):
    def __init__(self, integration_scheme="symplectic", dt=0.05, wrap_angle=False, **kwargs):
        super().__init__()
        self.integration_scheme = integration_scheme
        self.dt = dt
        self.wrap_angle = wrap_angle

    def derivative(self, state, action):
        raise NotImplementedError

    def get_Fa(self):
        raise NotImplementedError

    @staticmethod
    def wrap_state(state):
        wrapped = state.clone()
        wrapped[..., 0] = wrap_to_pi(state[..., 0])
        wrapped[..., 1] = wrap_to_pi(state[..., 1])
        return wrapped

    def forward(self, state, action):
        w_state = self.wrap_state(state) if self.wrap_angle else state

        if self.integration_scheme == "euler":
            return state + self.derivative(w_state, action) * self.dt
        elif self.integration_scheme == "rk4":
            func = partial(self.derivative, action=action)
            derivative = rk4_step_func_autonomous(func=func, dt=self.dt, y=w_state)
            new_state = state + derivative * self.dt
            return self.wrap_state(new_state)
        elif self.integration_scheme == "symplectic":

            assert state.shape[1] % 2 == 0  # [Only for x, dx] systems
            p = state.shape[1] // 2
            new_state = state + self.derivative(w_state, action) * self.dt  # Euler
            new_state[:, :p] = state[:, :p] + new_state[:, p:] * self.dt

            return new_state 


                
    def integrate(self, state, actions, resets=None):
        """
        Integrate a trajectory
        :param state: initial state, of shape batch x state size
        :param actions: sequence of actions, of shape: horizon x batch x action size
        :param resets: dict of (timestep, state) such that the trajectory at time timestep is reset to state
        :return: resulting trajectory, of shape: horizon x batch x state size
        """
        states = torch.zeros((actions.shape[0], state.shape[0], state.shape[1])).to(state.device)
        with torch.no_grad():
            for t in range(actions.shape[0]):
                state = self.forward(state, actions[t, ...])
                if resets and t in resets:
                    state = resets[t]
                states[t, ...] = state
        return states






class MLPModel_pendulum(DynamicsModel):
    def __init__(self, state_size, action_size, dt=0.05, hidden_size=16, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.state_size, self.action_size = state_size, action_size
        self.mlp = MLP(state_size + action_size, hidden_size, state_size)


    def derivative(self, state, action):
        """
            Predict dx_t = MLP(x_t,u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """

        xu = torch.cat((state, action), -1)
        return self.mlp(xu)

    def get_params(self):

        return {"gravity 15": 10.0}


    
class PerfectPendulumModel(DynamicsModel):
    def __init__(self, device="cuda", with_friction=True, integration_scheme="symplectic", dt=0.05, **kwargs):
        super().__init__(integration_scheme=integration_scheme, dt=dt, **kwargs)
        
        self.device = device
        
        self.g = nn.Parameter(torch.tensor(10.0).to(self.device), requires_grad=False)
        self.m = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=False)
        self.l = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=False)

        #self.gravity_norm = nn.Parameter(torch.tensor(15.0, dtype=torch.float).to(self.device), requires_grad=True)
        #self.inertia_norm = nn.Parameter(torch.tensor(3.0, dtype=torch.float).to(self.device), requires_grad=True)


        if with_friction:
            self.eta = nn.Parameter(torch.tensor(0.5).to(self.device), requires_grad=False)
            #self.friction_norm = nn.Parameter(torch.tensor(-0.5, dtype=torch.float).to(self.device), requires_grad=True)

        else:
            self.eta = nn.Parameter(torch.tensor(0.0).to(self.device), requires_grad=False)
            #self.friction_norm = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)

    def derivative(self, state, action):



        force = action * 2.0
        gravity_norm = 3 * self.g / (2 * self.l)
        friction_norm = - self.eta / (self.m * self.l**2)
        inertia_norm = 3.0 / (self.m * self.l**2)
        theta, thetadot = state[..., 0], state[..., 1]
        force = torch.clamp(force, -2.0, 2.0)

        dx = torch.zeros(state.shape).to(state.device)
        dx[:, 0] = thetadot
        dx[:, 1] = (gravity_norm) * torch.sin(theta) \
                   + (friction_norm) * thetadot \
                   + (inertia_norm) * force[:, 0]
        return dx

    def get_params(self):

        return {"gravity 10": self.g.data,
                "mass 1": self.m.data,
                "Lenght 1": self.l.data,
                "eta 0.5": self.eta.data}

class PendulumModel(DynamicsModel):
    def __init__(self, device="cuda", with_friction=True, integration_scheme="symplectic", dt=0.05, **kwargs):
        super().__init__(integration_scheme=integration_scheme, dt=dt, **kwargs)
        
        self.device = device
        
        self.g = nn.Parameter(torch.tensor(10.0).to(self.device), requires_grad=True)
        self.m = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=True)
        self.l = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=True)

        #self.gravity_norm = nn.Parameter(torch.tensor(15.0, dtype=torch.float).to(self.device), requires_grad=True)
        #self.inertia_norm = nn.Parameter(torch.tensor(3.0, dtype=torch.float).to(self.device), requires_grad=True)


        if with_friction:
            self.eta = nn.Parameter(torch.tensor(0.5).to(self.device), requires_grad=True)
            #self.friction_norm = nn.Parameter(torch.tensor(-0.5, dtype=torch.float).to(self.device), requires_grad=True)

        else:
            self.eta = nn.Parameter(torch.tensor(0.0).to(self.device), requires_grad=False)
            #self.friction_norm = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)

    def derivative(self, state, action):



        force = action * 2.0
        gravity_norm = 3 * self.g / (2 * self.l)
        friction_norm = - self.eta / (self.m * self.l**2)
        inertia_norm = 3.0 / (self.m * self.l**2)
        theta, thetadot = state[..., 0], state[..., 1]
        force = torch.clamp(force, -2.0, 2.0)


        
        
        dx = torch.zeros(state.shape).to(state.device)
        dx[:, 0] = thetadot
        dx[:, 1] = (gravity_norm) * torch.sin(theta) \
                   + (friction_norm) * thetadot \
                   + (inertia_norm) * force[:, 0]
        return dx

    def get_params(self):

        return {"gravity 10": self.g.data,
                "mass 1": self.m.data,
                "Lenght 1": self.l.data,
                "eta 0.5": self.eta.data}    

class FrictionlessPendulum(PendulumModel):
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        kwargs["with_friction"] = False
        super().__init__(**kwargs)


class AugmentedPendulumModel(DynamicsModel):
    """
    Frictionless pendulum
    """
    def __init__(self, device="cuda", integration_scheme="symplectic", dt=0.05, **kwargs):
        super().__init__(integration_scheme=integration_scheme, dt=dt, **kwargs)
        self.device = device
        self.pendulum = FrictionlessPendulum(**kwargs)
        self.mlp = MLPModel_pendulum(**kwargs)
        self.Fa = torch.tensor(0)
        self.Fp = torch.tensor(0)

    def derivative(self, state, action):
        self.Fp = self.pendulum.derivative(state, action)
        self.Fa = self.mlp.derivative(state, action)  # APHYNITY augmentation
        return self.Fp + self.Fa

    def get_Fa(self):
        return self.Fa, self.Fp


class Aphynity_pendulum(AugmentedPendulumModel):
    """
    Frictionless pendulum
    """
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        super().__init__(**kwargs)
        self.use_aphynity = True


    def get_params(self):

        return {"gravity 10": self.pendulum.g.data,
                "mass 1": self.pendulum.m.data,
                "Lenght 1": self.pendulum.l.data,
                "eta 0.5": self.pendulum.eta.data}    
    




class DynamicsModel_acrobot(nn.Module):
    def __init__(self, integration_scheme="symplectic", dt=0.2, wrap_angle=False, **kwargs):
        super().__init__()
        self.integration_scheme = integration_scheme
        self.dt = dt
        self.wrap_angle = wrap_angle

    def derivative(self, state, action):
        raise NotImplementedError

    def get_Fa(self):
        raise NotImplementedError

    @staticmethod
    def wrap_state(state):
        wrapped = state.clone()
        wrapped[..., 0] = wrap_to_pi(state[..., 0])
        wrapped[..., 1] = wrap_to_pi(state[..., 1])
        return wrapped

    def forward(self, state, action):
        w_state = self.wrap_state(state) if self.wrap_angle else state

        if self.integration_scheme == "euler":
            return state + self.derivative(w_state, action) * self.dt
        elif self.integration_scheme == "rk4":
            func = partial(self.derivative, action=action)
            derivative = rk4_step_func_autonomous(func=func, dt=self.dt, y=w_state)
            new_state = state + derivative * self.dt
            return self.wrap_state(new_state)
        elif self.integration_scheme == "symplectic":

            assert state.shape[1] % 2 == 0  # [Only for x, dx] systems
            p = state.shape[1] // 2
            new_state = state + self.derivative(w_state, action) * self.dt  # Euler
            new_state[:, :p] = state[:, :p] + new_state[:, p:] * self.dt

            return new_state #final_state


                
    def integrate(self, state, actions, resets=None):
        """
        Integrate a trajectory
        :param state: initial state, of shape batch x state size
        :param actions: sequence of actions, of shape: horizon x batch x action size
        :param resets: dict of (timestep, state) such that the trajectory at time timestep is reset to state
        :return: resulting trajectory, of shape: horizon x batch x state size
        """
        states = torch.zeros((actions.shape[0], state.shape[0], state.shape[1])).to(state.device)
        with torch.no_grad():
            for t in range(actions.shape[0]):
                state = self.forward(state, actions[t, ...])
                if resets and t in resets:
                    state = resets[t]
                states[t, ...] = state
        return states


class MLPModel_ctacrobot(DynamicsModel_acrobot):
    def __init__(self, state_size, action_size, hidden_size=16,  dt=0.2, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.state_size, self.action_size = state_size, action_size
        self.mlp = MLP_acr(state_size + action_size, hidden_size, state_size)

    def derivative(self, state, action):
        """
            Predict dx_t = MLP(x_t,u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """

        xu = torch.cat((state, action), -1)
        return self.mlp(xu)




       
class Perfectacrobot(DynamicsModel_acrobot):
    def __init__(self, device="cuda", with_friction=True, integration_scheme="symplectic", dt=0.2, **kwargs):
        super().__init__(integration_scheme=integration_scheme, dt=dt, **kwargs)
        self.device = device
        self.g = 9.8
        self.m1 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=False)
        self.m2 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=False)
        self.l1 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=False)
        self.l2 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=False)
        self.lc1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float).to(self.device), requires_grad=False)
        self.lc2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float).to(self.device), requires_grad=False)
        self.i1 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=False)
        self.i2 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=False)
        
        if with_friction:
            self.k1 = nn.Parameter(torch.tensor(0.01, dtype=torch.float).to(self.device), requires_grad=False)
            self.k2 = nn.Parameter(torch.tensor(0.05, dtype=torch.float).to(self.device), requires_grad=False)
        else:
            self.k1 = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)
            self.k2 = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)

    def derivative(self, state, action): 
        # state=[theta1 ; theta2 ; theta1_dot ; theta2_dot]
        #delta_angle = state[:,1]-state[:,0]
        alpha =  action[:,1] - self.k1 * state[:,2] 
        beta = action[:,0] - self.k2 * state[:,3] 

    
        dx = torch.zeros(state.shape).to(state.device)
        dx[:, 0] = state[:, 2]
        dx[:, 1] = state[:, 3]
        


        d1 = (
            self.m1 * (self.lc1)**2  + self.m2 * ( (self.l1**2) + (self.lc2)**2 + 2* self.l1 * self.lc2 * torch.cos(state[:,1]) )  + self.i1 + self.i2
            )

        d2 = self.m2 * ((self.lc2)**2 +  self.l1 * self.lc2 * torch.cos(state[:,1]))  + self.i2

        phi2 =  self.m2 * self.lc2 * self.g * torch.cos(state[:, 0] + state[:, 1] - math.pi/2.0)
        
        phi1 = (
            - self.m2 * self.l1 * self.lc2 * (state[:, 3])**2 * torch.sin(state[:, 1]) - 2* self.m2 * self.l1 * (self.lc2) * (state[:, 3]) * state[:, 2] * torch.sin(state[:, 1]) + (self.m1 * (self.lc1) + self.m2 * self.l1) * self.g * torch.cos(state[:,0] - math.pi/2.0)  + self.m2 * self.lc2 * self.g * torch.cos(state[:, 0] + state[:, 1] - math.pi/2.0)
        )
        
        dx[:, 3] =   (beta + d2/d1 * phi1 - self.m2 * self.l1 * self.lc2 * state[:, 2]**2 * torch.sin(state[:, 1]) - phi2) / (self.m2 * self.lc2**2 + self.i2 - d2**2/ d1) 

        dx[:, 2] = - (alpha + d2 * ((beta + d2/d1 * phi1 - self.m2 * self.l1 * self.lc2 * state[:, 2]**2 * torch.sin(state[:, 1]) - phi2) / (self.m2 * self.lc2**2 + self.i2 - d2**2/ d1)) + phi1 )  / d1

        return dx     



    def get_params(self):

        return {"m1": self.m1.data,
                "m2": self.m2.data,
                "l1": self.l1.data,
                "l2": self.l2.data,
                "lc1": self.lc1.data,
                "lc2": self.lc2.data,
                "I1": self.i1.data,
                "I2": self.i2.data,
                "K1 3": self.k1.data,
                "K2 0.5": self.k2.data}
       

       
class ctacrobotModel(DynamicsModel_acrobot):
    def __init__(self, device="cuda", with_friction=True, integration_scheme="symplectic", dt=0.2, **kwargs):
        super().__init__(integration_scheme=integration_scheme, dt=dt, **kwargs)
        self.device = device
        self.g = 9.8
        self.m1 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=True)
        self.m2 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=True)
        self.l1 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=True)
        self.l2 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=True)
        self.lc1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float).to(self.device), requires_grad=True)
        self.lc2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float).to(self.device), requires_grad=True)
        self.i1 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=True)
        self.i2 = nn.Parameter(torch.tensor(1, dtype=torch.float).to(self.device), requires_grad=True)
        
        if with_friction:
            self.k1 = nn.Parameter(torch.tensor(0.01, dtype=torch.float).to(self.device), requires_grad=True)
            self.k2 = nn.Parameter(torch.tensor(0.05, dtype=torch.float).to(self.device), requires_grad=True)
        else:
            self.k1 = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)
            self.k2 = nn.Parameter(torch.tensor(0, dtype=torch.float).to(self.device), requires_grad=False)

    def derivative(self, state, action): 
        # state=[theta1 ; theta2 ; theta1_dot ; theta2_dot]
        #delta_angle = state[:,1]-state[:,0]
        
        alpha = action[:,1]  - self.k1 * state[:,2] 
        beta = action[:,0] - self.k2 * state[:,3] 

    
        dx = torch.zeros(state.shape).to(state.device)
        dx[:, 0] = state[:, 2]
        dx[:, 1] = state[:, 3]
        


        d1 = (
            self.m1 * (self.lc1)**2  + self.m2 * ( (self.l1**2) + (self.lc2)**2 + 2* self.l1 * self.lc2 * torch.cos(state[:,1]) )  + self.i1 + self.i2
            )


        d2 = self.m2 * ((self.lc2)**2 +  self.l1 * self.lc2 * torch.cos(state[:,1]))  + self.i2
        
        phi2 =  self.m2 * self.lc2 * self.g * torch.cos(state[:, 0] + state[:, 1] - math.pi/2.0)
        
        phi1 = (
            - self.m2 * self.l1 * self.lc2 * (state[:, 3])**2 * torch.sin(state[:, 1]) 
            - 2* self.m2 * self.l1 * (self.lc2) * (state[:, 3]) * state[:, 2] * torch.sin(state[:, 1]) 
            + (self.m1 * (self.lc1) + self.m2 * self.l1) * self.g * torch.cos(state[:,0] - math.pi/2.0) 
              + phi2
        )
        
        dx[:, 3] =   (beta + d2/d1 * phi1 - self.m2 * self.l1 * self.lc2 * state[:, 2]**2 * torch.sin(state[:, 1]) - phi2) / (self.m2 * self.lc2**2 + self.i2 - d2**2/ d1) 

        dx[:, 2] = - (alpha + d2 * dx[:, 3] + phi1 )  / d1

        return dx     



    def get_params(self):

        return {"m1": self.m1.data,
                "m2": self.m2.data,
                "l1": self.l1.data,
                "l2": self.l2.data,
                "lc1": self.lc1.data,
                "lc2": self.lc2.data,
                "I1": self.i1.data,
                "I2": self.i2.data,
                "K1 3": self.k1.data,
                "K2 0.5": self.k2.data}
       
class Frictionlessctacrobot(ctacrobotModel):
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        kwargs["with_friction"] = False
        super().__init__(**kwargs)
        
        

class AugmentedDoublePendulumModel(DynamicsModel_acrobot):
    """
    Frictionless pendulum
    """
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        super().__init__(integration_scheme=integration_scheme, **kwargs)
        self.device = device
        self.pendulum = Frictionlessctacrobot(**kwargs)
        self.mlp = MLPModel_ctacrobot(**kwargs)
        self.Fa = torch.tensor(0)
        self.Fp = torch.tensor(0)

    def derivative(self, state, action):
        self.Fp = self.pendulum.derivative(state, action)
        self.Fa = self.mlp.derivative(state, action)  # APHYNITY augmentation
        return self.Fp + self.Fa

    def get_Fa(self):
        return self.Fa, self.Fp


class Aphynity_ctacrobot(AugmentedDoublePendulumModel):
    """
    Frictionless pendulum
    """
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        super().__init__(**kwargs)
        self.use_aphynity = True
        

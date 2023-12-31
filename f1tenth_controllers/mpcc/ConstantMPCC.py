import numpy as np 
import matplotlib.pyplot as plt
import casadi as ca

from f1tenth_controllers.mpcc.ReferencePath import ReferencePath
from f1tenth_controllers.mpcc.mpcc_utils import *

VERBOSE = False
# VERBOSE = True


NX = 4
NU = 2

class ConstantMPCC:
    def __init__(self, map_name):
        self.params = load_mpcc_params()
        self.rp = ReferencePath(map_name, 0.25)
        self.N = self.params.N

        self.u0 = np.zeros((self.N, NU))
        self.X0 = np.zeros((self.N + 1, NX))
        self.warm_start = True # warm start every time

        self.init_optimisation()
        self.init_constraints()
        self.init_bounds()
        self.init_objective()
        self.init_solver()
       
    def init_optimisation(self):
        states = ca.MX.sym('states', NX) #[x, y, psi, s]
        controls = ca.MX.sym('controls', NU) # [delta, p]

        rhs = ca.vertcat(self.params.vehicle_speed * ca.cos(states[2]), self.params.vehicle_speed * ca.sin(states[2]), (self.params.vehicle_speed / self.params.wheelbase) * ca.tan(controls[0]), controls[1])  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        
        self.U = ca.MX.sym('U', NU, self.N)
        self.X = ca.MX.sym('X', NX, (self.N + 1))
        self.P = ca.MX.sym('P', NX + 2 * self.N) # init state and boundaries of the reference path

    def init_constraints(self):
        '''Initialize upper and lower bounds for state and control variables'''
        self.lbg = np.zeros((NX * (self.N + 1) + self.N, 1))
        self.ubg = np.zeros((NX * (self.N + 1) + self.N, 1))
        self.lbx = np.zeros((NX + (NX + NU) * self.N, 1))
        self.ubx = np.zeros((NX + (NX + NU) * self.N, 1))
                
        x_min, y_min = np.min(self.rp.path, axis=0) - 2
        x_max, y_max = np.max(self.rp.path, axis=0) + 2
        s_max = self.rp.s_track[-1] *1.5
        lbx = np.array([[x_min, y_min, self.params.psi_min, 0]])
        ubx = np.array([[x_max, y_max, self.params.psi_max, s_max]])
        for k in range(self.N + 1):
            self.lbx[NX * k:NX * (k + 1), 0] = lbx
            self.ubx[NX * k:NX * (k + 1), 0] = ubx

        state_count = NX * (self.N + 1)
        for k in range(self.N):
            self.lbx[state_count:state_count + NU, 0] = np.array([[-self.params.delta_max, self.params.p_min]]) 
            self.ubx[state_count:state_count + NU, 0] = np.array([[self.params.delta_max, self.params.p_max]])  
            state_count += NU

    def init_bounds(self):
        """Initialise the bounds (g) on the dynamics and track boundaries"""
        self.g = self.X[:, 0] - self.P[:NX]  # initial condition constraints
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            k1 = self.f(self.X[:, k], self.U[:, k])
            st_next_euler = self.X[:, k] + (self.params.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # add dynamics constraint

            self.g = ca.vertcat(self.g, self.P[NX + 2 * k] * st_next[0] - self.P[NX + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  :represents path boundary constraints

    def init_objective(self):
        self.obj = 0  # Objective function
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            t_angle = self.rp.angle_lut_t(st_next[3])
            ref_x, ref_y = self.rp.center_lut_x(st_next[3]), self.rp.center_lut_y(st_next[3])
            countour_error = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            lag_error = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.obj = self.obj + countour_error **2 * self.params.weight_contour  
            self.obj = self.obj + lag_error **2 * self.params.weight_lag
            self.obj = self.obj - self.U[1, k] * self.params.weight_progress 
            self.obj = self.obj + (self.U[0, k]) ** 2 * self.params.weight_steer 
            
    def init_solver(self):
        optimisation_variables = ca.vertcat(ca.reshape(self.X, NX * (self.N + 1), 1),
                                ca.reshape(self.U, NU * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': optimisation_variables, 'g': self.g, 'p': self.P}
        opts = {"ipopt": {"max_iter": 2000, "print_level": 0}, "print_time": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def plan(self, obs):
        x0 = self.build_initial_state(obs)
        self.construct_warm_start_soln(x0) 
        self.set_up_constraints()
        p = self.generate_parameters(x0)
        controls = self.solve(p)

        action = np.array([controls[0, 0], self.params.vehicle_speed])

        return action 

    def build_initial_state(self, obs):
        x0 = obs["vehicle_state"][np.array([0, 1, 4])]
        x0[2] = normalise_psi(x0[2]) 
        x0 = np.append(x0, self.rp.calculate_s(x0[0:2]))

        return x0

    def generate_parameters(self, x0_in):
        p = np.zeros(NX + 2 * self.N)
        p[:NX] = x0_in

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s_progress = self.X0[k, 3]
            right_x = self.rp.right_lut_x(s_progress).full()[0, 0]
            right_y = self.rp.right_lut_y(s_progress).full()[0, 0]
            left_x = self.rp.left_lut_x(s_progress).full()[0, 0]
            left_y = self.rp.left_lut_y(s_progress).full()[0, 0]

            delta_x = right_x - left_x
            delta_y = right_y - left_y
            p[NX + 2 * k:NX + 2 * k + 2] = [-delta_x, delta_y]

        return p
    
    def set_up_constraints(self):
        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s_progress = self.X0[k, 3]
            right_x = self.rp.right_lut_x(s_progress).full()[0, 0]
            right_y = self.rp.right_lut_y(s_progress).full()[0, 0]
            left_x = self.rp.left_lut_x(s_progress).full()[0, 0]
            left_y = self.rp.left_lut_y(s_progress).full()[0, 0]

            delta_x = right_x - left_x
            delta_y = right_y - left_y

            self.lbg[NX - 1 + (NX + 1) * (k + 1), 0] = min(-delta_x * right_x - delta_y * right_y,
                                    -delta_x * left_x - delta_y * left_y) 
            self.ubg[NX - 1 + (NX + 1) * (k + 1), 0] = max(-delta_x * right_x - delta_y * right_y,
                                    -delta_x * left_x - delta_y * left_y)

        self.lbg[NX *2, 0] = - ca.inf
        self.ubg[NX *2, 0] = ca.inf

    def solve(self, p):
        x_init = ca.vertcat(ca.reshape(self.X0.T, NX * (self.N + 1), 1),
                         ca.reshape(self.u0.T, NU * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        self.X0 = ca.reshape(sol['x'][0:NX * (self.N + 1)], NX, self.N + 1).T
        controls = ca.reshape(sol['x'][NX * (self.N + 1):], NU, self.N).T

        if self.solver.stats()['return_status'] != 'Solve_Succeeded':
            print("Solve failed!!!!!")

        return controls.full()
        
    def construct_warm_start_soln(self, initial_state):
        if not self.warm_start: return
        # self.warm_start = False

        self.X0 = np.zeros((self.N + 1, NX))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.params.p_initial * self.params.dt

            psi_next = self.rp.angle_lut_t(s_next).full()[0, 0]
            x_next, y_next = self.rp.center_lut_x(s_next), self.rp.center_lut_y(s_next)

            # adjusts the centerline angle to be continuous
            psi_diff = self.X0[k-1, 2] - psi_next
            psi_mul = self.X0[k-1, 2] * psi_next
            if (abs(psi_diff) > np.pi and psi_mul < 0) or abs(psi_diff) > np.pi*1.5:
                if psi_diff > 0:
                    psi_next += np.pi * 2
                else:
                    psi_next -= np.pi * 2
            self.X0[k, :] = np.array([x_next.full()[0, 0], y_next.full()[0, 0], psi_next, s_next])




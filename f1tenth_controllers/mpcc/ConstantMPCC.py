import numpy as np 
import matplotlib.pyplot as plt
import casadi as ca

from f1tenth_controllers.mpcc.TrackLine import TrackLine
from f1tenth_controllers.mpcc.ReferencePath import ReferencePath

VERBOSE = False
# VERBOSE = True
SPEED = 2 # m/s
L = 0.33
WIDTH = 0.3 # m on each side

WEIGHT_PROGRESS = 0.0
WEIGHT_LAG = 500
WEIGHT_CONTOUR = 0.1
WEIGHT_STEER = 1000

np.printoptions(precision=2, suppress=True)


def normalise_psi(psi):
    while psi > np.pi:
        psi -= 2*np.pi
    while psi < -np.pi:
        psi += 2*np.pi
    return psi


class ConstantMPCC:
    def __init__(self, map_name):
        self.rp = ReferencePath(map_name, 0.25)
        self.dt = 0.2
        self.N = 10 # number of steps to predict
        self.nx = 4
        self.nu = 2
        self.track = TrackLine(map_name, False)
        self.p_initial = 2
        
        self.x_min, self.y_min = np.min(self.rp.path, axis=0) - 2
        self.psi_min, self.s_min = -100, 0
        self.x_max, self.y_max = np.max(self.rp.path, axis=0) + 2
        self.psi_max, self.s_max = 100, self.rp.s_track[-1] *1.5

        self.delta_min, self.p_min = -0.4, 0
        self.delta_max, self.p_max = 0.4, 4

        self.u0 = np.zeros((self.N, self.nu))
        self.X0 = np.zeros((self.N + 1, self.nx))
        self.warm_start = True # warm start every time

        self.init_optimisation()
        self.init_constraints()
        self.init_objective()
        self.init_solver()
       
    def init_optimisation(self):
        # States
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        psi = ca.MX.sym('psi')
        s = ca.MX.sym('s')
        # Controls
        # v = ca.MX.sym('v')
        delta = ca.MX.sym('delta')
        p = ca.MX.sym('p')

        states = ca.vertcat(x, y, psi, s)
        controls = ca.vertcat(delta, p)
        rhs = ca.vertcat(SPEED * ca.cos(psi), SPEED * ca.sin(psi), (SPEED / L) * ca.tan(delta), p)  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        self.U = ca.MX.sym('U', self.nu, self.N)
        self.X = ca.MX.sym('X', self.nx, (self.N + 1))
        self.P = ca.MX.sym('P', self.nx + 2 * self.N) # init state and boundaries of the reference path

    def init_constraints(self):
        '''Initialize constraints for states, dynamic model state transitions and control inputs of the system'''
        self.lbg = np.zeros((self.nx * (self.N + 1) + self.N, 1))
        self.ubg = np.zeros((self.nx * (self.N + 1) + self.N, 1))
        self.lbx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
        self.ubx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
        # Upper and lower bounds for the state optimization variables
        lbx = np.array([[self.x_min, self.y_min, self.psi_min, self.s_min]])
        ubx = np.array([[self.x_max, self.y_max, self.psi_max, self.s_max]])
        for k in range(self.N + 1):
            self.lbx[self.nx * k:self.nx * (k + 1), 0] = lbx
            self.ubx[self.nx * k:self.nx * (k + 1), 0] = ubx
        state_count = self.nx * (self.N + 1)
        # Upper and lower bounds for the control optimization variables
        for k in range(self.N):
            self.lbx[state_count:state_count + self.nu, 0] = np.array([[self.delta_min, self.p_min]])  # v and delta lower bound
            self.ubx[state_count:state_count + self.nu, 0] = np.array([[self.delta_max, self.p_max]])  # v and delta upper bound
            state_count += self.nu

    def init_objective(self):
        self.obj = 0  # Objective function
        self.g = []  # constraints vector

        st = self.X[:, 0]  # initial state
        self.g = ca.vertcat(self.g, st - self.P[:self.nx])  # initial condition constraints
        for k in range(self.N):
            st = self.X[:, k]
            st_next = self.X[:, k + 1]
            con = self.U[:, k]
            
            t_angle = self.rp.angle_lut_t(st_next[3])
            ref_x, ref_y = self.rp.center_lut_x(st_next[3]), self.rp.center_lut_y(st_next[3])
            #Contouring error
            e_c = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            #Lag error
            e_l = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.obj = self.obj + e_c **2 * WEIGHT_CONTOUR  
            self.obj = self.obj + e_l **2 * WEIGHT_LAG
            self.obj = self.obj - con[1] * WEIGHT_PROGRESS 
            self.obj = self.obj + (con[0]) ** 2 * WEIGHT_STEER  # minimize the use of steering input


            k1 = self.f(st, con)
            st_next_euler = st + (self.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # compute constraints

            # path boundary constraints
            self.g = ca.vertcat(self.g, self.P[self.nx + 2 * k] * st_next[0] - self.P[self.nx + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  --represents half space planes

            

    def init_solver(self):
        opts = {}
        opts["ipopt"] = {}
        opts["ipopt"]["max_iter"] = 2000
        opts["ipopt"]["print_level"] = 0
        opts["print_time"] = 0
        
        OPT_variables = ca.vertcat(ca.reshape(self.X, self.nx * (self.N + 1), 1),
                                ca.reshape(self.U, self.nu * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}
        # self.solver = ca.nlpsol('solver', nlp_prob)
        # self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob)
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def plan(self, obs):
        x0 = obs["vehicle_state"][np.array([0, 1, 4])]
        x0[2] = normalise_psi(x0[2]) 

        x0 = np.append(x0, self.rp.calculate_s(x0[0:2]))
        x0[3] = self.filter_estimate(x0[3])

        self.warm_start = True
        p = self.generate_constraints_and_parameters(x0)
        states, controls, solved_status = self.solve(p)
        # if not solved_status:
        #     self.warm_start = True
        #     p = self.generate_constraints_and_parameters(x0)
        #     states, controls, solved_status = self.solve(p)
        #     if VERBOSE:
        #         print(f"Solve failed: ReWarm Start: New outcome: {solved_status}")

        s = states[:, 3]
        s = [s[k] if s[k] < self.rp.track_length else s[k] - self.rp.track_length for k in range(self.N+1)]
        c_pts = [[self.rp.center_lut_x(states[k, 3]).full()[0, 0], self.rp.center_lut_y(states[k, 3]).full()[0, 0]] for k in range(self.N + 1)]

        if VERBOSE:
            self.rp.plot_path()
            plt.figure(2)
            plt.plot(states[:, 0], states[:, 1], 'r--')
            for i in range(self.N + 1):
                xs = [states[i, 0], c_pts[i][0]]
                ys = [states[i, 1], c_pts[i][1]]
                plt.plot(xs, ys, '--', color='orange')

            size = 9
            plt.xlim([x0[0] - size, x0[0] + size])
            plt.ylim([x0[1] - size, x0[1] + size])
            plt.pause(0.001)

            if not solved_status:
                plt.show()

        first_control = controls[0, :]
        steering_angle = first_control[0]
        action = np.array([steering_angle, 2])

        return action # return the first control action

    def generate_constraints_and_parameters(self, x0_in):
        if self.warm_start:
            if VERBOSE:
                print(f"Warm starting with condition: {x0_in}")
            self.construct_warm_start_soln(x0_in) 

        p = np.zeros(self.nx + 2 * self.N)
        p[:self.nx] = x0_in

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s = self.X0[k, 3]
            if s > self.rp.track_length:
                s = s - self.rp.track_length
            right_point = [self.rp.right_lut_x(s).full()[0, 0], self.rp.right_lut_y(s).full()[0, 0]]
            left_point = [self.rp.left_lut_x(s).full()[0, 0], self.rp.left_lut_y(s).full()[0, 0]]

            delta_x_path = right_point[0] - left_point[0]
            delta_y_path = right_point[1] - left_point[1]
            p[self.nx + 2 * k:self.nx + 2 * k + 2] = [-delta_x_path, delta_y_path]

            up_bound = max(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
                           -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            low_bound = min(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
                            -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            self.lbg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = low_bound # check this, there could be an error
            self.ubg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = up_bound

        # the optimizer cannot control the init state.
        self.lbg[self.nx *2, 0] = - ca.inf
        self.ubg[self.nx *2, 0] = ca.inf

        return p

    def solve(self, p):
        x_init = ca.vertcat(ca.reshape(self.X0.T, self.nx * (self.N + 1), 1),
                         ca.reshape(self.u0.T, self.nu * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        # Get state and control solution
        self.X0 = ca.reshape(sol['x'][0:self.nx * (self.N + 1)], self.nx, self.N + 1).T  # get soln trajectory
        u = ca.reshape(sol['x'][self.nx * (self.N + 1):], self.nu, self.N).T  # get controls solution

        trajectory = self.X0.full()  # size is (N+1,n_states)
        inputs = u.full()
        stats = self.solver.stats()
        solved_status = True
        if stats['return_status'] != 'Solve_Succeeded':
            # print(stats['return_status'])
            solved_status = False

        # Shift trajectory and control solution to initialize the next step
        self.X0 = ca.vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = ca.vertcat(u[1:, :], u[u.size1() - 1, :])

        return trajectory, inputs, solved_status
        
    def filter_estimate(self, initial_arc_pos):
        if (self.X0[0, 3] >= self.rp.track_length) and (
                (initial_arc_pos >= self.rp.track_length) or (initial_arc_pos <= 5)):
            self.X0[:, 3] = self.X0[:, 3] - self.rp.track_length
        if initial_arc_pos >= self.rp.track_length:
            initial_arc_pos -= self.rp.track_length
        return initial_arc_pos

    def construct_warm_start_soln(self, initial_state):
        self.X0 = np.zeros((self.N + 1, self.nx))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.p_initial * self.dt
            if s_next > self.rp.track_length:
                s_next = s_next - self.rp.track_length

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

        self.warm_start = False

    def done_callback(self, final_obs):
        progress = self.track.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Lap complete ({self.track.map_name.upper()}) --> Time: {final_obs['lap_times'][0]:.2f}, Progress: {progress:.1f}%")


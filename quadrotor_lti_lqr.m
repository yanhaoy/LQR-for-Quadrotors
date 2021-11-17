clear all; clc; close all;

addpath('utils\')
addpath('casadi\');
import casadi.*

%%
par.T = 5; % Time horizon
par.dt = 0.002; % Main control loop runs in 500 Hz
par.m = 28*1e-3; % Mass
par.g = 9.81; % Gravity
par.J = [16.571710, 0.830806, 0.718277;
    0.830806, 16.655602, 1.800197;
    0.718277, 1.800197, 29.261652]*1e-6; % Inertia
par.tau_k = 0.005964552; % Mapping from thrust to torque
par.l_x = 28.243*1e-3; % Moment arm in x direction
par.l_y = 28.325*1e-3; % Moment arm in y direction
par.Q = inv(diag([0.5, 0.38, 0.38, 0.38, 4*0.5, 4*0.38, 4*0.38, 4*0.38].^2)); % LQR gains
par.R = inv((2*0.06867)^2*eye(4)); % LQR gains

[model] = dynamics(par);
[K] = lti_lqr(model, par);

%%
% Simulation test
X = [];
U = [];
xk = ones(12, 1)*1e-3;
xd = [0.1;0;0;0;0;0;0;0];
X(:, 1) = xk;
for k=0:500
    uk = repmat(28*1e-3*9.81/4, 4, 1) - K*(xk([3:6, 9:12])-xd);
    U(:, k+1) = uk;
    
    sol = model.dyn_sim('x0', xk, 'p', [uk]);
    xk = full(sol.xf);
    X(:, k+2) = xk;
end

function [x, xdot, tau] = quadrotor_dynamics(par)

import casadi.*

% Declare model variables
x = SX.sym('x', 12, 1);
tau = SX.sym('Y', 4, 1);

[m, g, J, l_x, l_y, tau_k] = deal(par.m, par.g, par.J, par.l_x, par.l_y, par.tau_k);
[px, py, pz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz] = ...
    deal(x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11), x(12)); % Position, velocity in world frame, angular velocity in body frame, pitch rotates clock-wise
[F, tau_x, tau_y, tau_z] = deal(sum(tau), ...
[-l_x, -l_x, l_x, l_x]*tau, ...
[-l_y, l_y, l_y, -l_y]*tau, ...
[-tau_k, tau_k, -tau_k, tau_k]*tau); % Thrust and torque in body frame

% Rotation matrix
R_wb = [cos(yaw), -sin(yaw), 0;
    sin(yaw), cos(yaw), 0;
    0, 0, 1]*...
    [cos(pitch), 0, -sin(pitch);
    0, 1, 0;
    sin(pitch), 0, cos(pitch)]*...
    [1, 0, 0;
    0, cos(roll), -sin(roll);
    0, sin(roll), cos(roll)]; % Pitch rotate in clock-wise

% Body Jacobian
J_wb_b = jacobian(reshape(R_wb, 9, 1), [roll; pitch; yaw]);
J_wb_b = [skew2angvel(R_wb'*reshape(J_wb_b(:, 1), 3, 3)), ...
    skew2angvel(R_wb'*reshape(J_wb_b(:, 2), 3, 3)), ...
    skew2angvel(R_wb'*reshape(J_wb_b(:, 3), 3, 3))];

% Newton-Euler equation
M = blkdiag(diag(repmat(m, 3, 1)), J); % Mass matrix
h = [0; 0; m*g; cross([wx; wy; wz], J*[wx; wy; wz])]; % Nonlinear terms
Y = [R_wb*[0; 0; F]; tau_x; tau_y; tau_z]; % Input in generalized coordinates

% Euqation of motion
xdot = [vx; vy; vz; J_wb_b\[wx; wy; wz]; M\(Y-h)];

end

function [model] = dynamics(par)

import casadi.*

dt = par.dt;

[x, xdot, u] = quadrotor_dynamics(par);

% Objective term
L = x'*x + u'*u;

% Continuous time Jacobian
A = jacobian(xdot, x);
B = jacobian(xdot, u);
model.jac = Function('jacobian', {x, u}, {A, B}, {'x', 'u'}, {'A', 'B'});

% CVODES from the SUNDIALS suite
dae = struct('x',x,'p',[u],'ode',xdot,'quad',L);
opts = struct('tf',dt);
model.dyn_sim = integrator('F', 'cvodes', dae, opts);

end

function [K] = lti_lqr(model, par)

dt = par.dt;
jac = model.jac;
Q = par.Q;
R = par.R;

x = zeros(12, 1);
u = repmat(par.m*par.g/4, 4, 1); % Hovering input
        
sol = jac('x', x, 'u', u);
A = full(sol.A);
B = full(sol.B);

sys = ss(A, B, eye(12), zeros(12, 4));
sysd = c2d(sys, dt, 'ZOH');
[Ad, Bd, ~, ~] = ssdata(sysd);

% Only consider height and orientation
Ad = Ad([3:6, 9:12], [3:6, 9:12]);
Bd = Bd([3:6, 9:12], :);

[K, ~, ~] = dlqr(Ad, Bd, Q, R);

end

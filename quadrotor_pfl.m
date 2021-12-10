clear all; clc; close all;

addpath('utils\')
addpath('casadi\');
import casadi.*

%%
par.T = 5; % Time horizon
par.dt = 0.002; % Main control loop runs in 500 Hz
par.m = 27*1e-3; % Mass
par.g = 9.81; % Gravity
% par.J = [16.571710, 0.830806, 0.718277;
%     0.830806, 16.655602, 1.800197;
%     0.718277, 1.800197, 29.261652]*1e-6; % Inertia
par.tau_k = 0.005964552; % Mapping from thrust to torque
par.l_x = 0.046*cos(pi/4); % Moment arm in x direction
par.l_y = 0.046*sin(pi/4); % Moment arm in y direction
par.l_z = 0.024;
par.J=diag([1.657171e-05; 1.657171e-05; 2.9261652e-05]+...
    4*5e-4*...
    [par.l_y^2+par.l_z^2;
    par.l_x^2+par.l_z^2;
    par.l_x^2+par.l_y^2]); % Body inertia
par.exp_data = [
    0.05423;
    0.05423; 
    0.03792
    0.007495;
    0.007495;
    0.01911;
    0.02342;
    0.02342;
    0.03169;
    0.1441;
    0.1441;
    0.03733];
par.Q = diag([
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    ]); % LQR gains
par.R = 1*eye(4); % LQR gains

%%
[Kpd] = pdLQR(par);

%%
[model] = quadrotor_dynamics(par);
[K] = lti_lqr(model, par);

%%
function [Kpd] = pdLQR(par)

[dt, m, g, J, l_x, l_y, tau_k] = deal(par.dt, par.m, par.g, par.J, par.l_x, par.l_y, par.tau_k);

s = tf('s');
sys = c2d(1/s^2, dt, 'ZOH');
Crp = pidtune(sys,'PD',500*2*pi/60,pidtuneOptions('DesignFocus','disturbance-rejection','PhaseMargin',90));
% figure()
% stepplot(feedback(sys*Crp, 1))
sys = c2d(1/s^2, dt, 'ZOH');
Cy = pidtune(sys,'PD',500*2*pi/60,pidtuneOptions('DesignFocus','disturbance-rejection','PhaseMargin',90));
% figure()
% stepplot(feedback(sys*Cy, 1))
sys = c2d(1/s^2, dt*5, 'ZOH');
Cz = pidtune(sys,'PD',100*2*pi/90,pidtuneOptions('DesignFocus','disturbance-rejection','PhaseMargin',70));
% figure()
% stepplot(feedback(sys*Cz, 1))
s = tf('s');
sys = c2d(1/s^2, dt*5, 'ZOH');
Cxy = pidtune(sys,'PD',100*2*pi/30,pidtuneOptions('DesignFocus','disturbance-rejection','PhaseMargin',80));

Kpd = zeros(4, 12);
[Kpd(2, 4), ~, Kpd(2, 10)] = piddata(Crp);
[Kpd(3, 5), ~, Kpd(3, 11)] = piddata(Crp);
[Kpd(4, 6), ~, Kpd(4, 12)] = piddata(Cy);
[Kpd(1, 3), ~, Kpd(1, 9)] = piddata(Cz);

Kpd2 = zeros(4, 12);
[Kpd2(2, 4), ~, Kpd2(2, 10)] = piddata(Cxy);
[Kpd2(3, 5), ~, Kpd2(3, 11)] = piddata(Cxy);
[Kpd2(4, 6), ~, Kpd2(4, 12)] = piddata(Cy);
[Kpd2(1, 3), ~, Kpd2(1, 9)] = piddata(Cz);

Kpd(:, 1) = -Kpd2(:, 5)*0.0934;
Kpd(:, 2) = -Kpd2(:, 4)*0.0934;
Kpd(:, 7) = -Kpd2(:, 11)*0.3188;
Kpd(:, 8) = -Kpd2(:, 10)*0.3188;

end

function [model] = quadrotor_dynamics(par)

syms x [12, 1] real % x, y, z, r, p, y, dx, dy, dz, dr, dp, dy
syms u [4, 1] real % ddz, ddr, ddp, ddy

[m, g, J, l_x, l_y, tau_k] = deal(par.m, par.g, par.J, par.l_x, par.l_y, par.tau_k);
[roll, pitch, yaw] = deal(x(4), x(5), x(6));

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
J_wb_b = jacobian(reshape(R_wb, 9, 1), x(4:6));
J_wb_b = [skew2angvel(R_wb'*reshape(J_wb_b(:, 1), 3, 3)), ...
    skew2angvel(R_wb'*reshape(J_wb_b(:, 2), 3, 3)), ...
    skew2angvel(R_wb'*reshape(J_wb_b(:, 3), 3, 3))];
J_wb_b = elementwiseSimplify(J_wb_b);
matlabFunction(J_wb_b, 'File', 'compJ', 'Vars', {x(4:6)});

w = J_wb_b*x(10:12);
J_dot = reshape(jacobian(reshape(J_wb_b, 9, 1), x(4:6))*x(10:12), 3, 3);

% Newton-Euler equation
M = sym(blkdiag(diag(repmat(m, 3, 1)), J)); % Mass matrix
h = [0; 0; m*g; cross(w, J*w)]; % Nonlinear terms

h = h + M*[zeros(3, 1); J_dot*x(10:12)];
M = M*blkdiag(eye(3), J_wb_b);

T = [1,1,1,1;
    -l_x, -l_x, l_x, l_x;
    -l_y, l_y, l_y, -l_y;
    -tau_k, tau_k, -tau_k, tau_k];

M = blkdiag(eye(2), inv(T))*blkdiag(R_wb', eye(3))*M;
h = blkdiag(eye(2), inv(T))*blkdiag(R_wb', eye(3))*h;

M_11 = M(1:2, 1:2);
M_12 = M(1:2, 3:end);
M_21 = M(3:end, 1:2);
M_22 = M(3:end, 3:end);
h_1 = h(1:2);
h_2 = h(3:end);

tau = (M_22 - M_21*inv(M_11)*M_12)*u + h_2 - M_21*inv(M_11)*h_1;
tau = elementwiseSimplify(tau);
matlabFunction(tau, 'File', 'compTau', 'Vars', {x, u});

x_dot = [x(7:12); M_11\(-h_1-M_12*u); u];
x_dot = elementwiseSimplify(x_dot);

model.A = double(subs(jacobian(x_dot, x), [x; u], [zeros(12, 1); zeros(4, 1)]));
model.B = double(subs(jacobian(x_dot, u), [x; u], [zeros(12, 1); zeros(4, 1)]));

end

function [K] = lti_lqr(model, par)

dt = par.dt;
Q = par.Q;
R = par.R;

A = model.A;
B = model.B;

sys = ss(A, B, eye(12), zeros(12, 4));
sysd = c2d(sys, dt, 'ZOH');
[Ad, Bd, ~, ~] = ssdata(sysd);

[K, ~, ~] = dlqr(Ad, Bd, Q, R);

end

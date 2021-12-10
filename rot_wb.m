function [x] = rot_wb(x, rpy, inv)

[roll, pitch, yaw] = deal(rpy(1), rpy(2), rpy(3));

R_wb = [cos(yaw), -sin(yaw), 0;
    sin(yaw), cos(yaw), 0;
    0, 0, 1]*...
    [cos(pitch), 0, -sin(pitch);
    0, 1, 0;
    sin(pitch), 0, cos(pitch)]*...
    [1, 0, 0;
    0, cos(roll), -sin(roll);
    0, sin(roll), cos(roll)];

if inv
    x = R_wb*x;
else
    x = R_wb'*x;
end

end
function F = theta_interp(Th, X0, Y0)
% Axial-safe interpolant of channel azimuth theta(x) [deg], via (cos2t, sin2t).
xk = (Th.X_GK - X0)/1000; yk = (Th.Y_GK - Y0)/1000;
t2 = deg2rad(2*Th.channel_azimuth_deg);
Fc = scatteredInterpolant(xk, yk, cos(t2), 'natural', 'nearest');
Fs = scatteredInterpolant(xk, yk, sin(t2), 'natural', 'nearest');
F  = @(X) rad2deg( atan2(Fs(X(1,:)',X(2,:)'), Fc(X(1,:)',X(2,:)')) / 2 );
end

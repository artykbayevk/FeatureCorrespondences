function [ threeDcoordinates ] = findThreeDcoordinates( distortedLeftCoordinates, distortedRightCoordinates  )

fc_left = [ 767.48024   764.21925 ] ;
cc_left = [ 314.12348   205.28206 ] ;
alpha_c_left = [ 0.00000 ] ;
kc_left = [ -0.12140   0.02189   0.00747   0.00224  0.00000 ] ;


fc_right = [ 779.92851   777.71863 ] ;
cc_right = [ 287.73953   188.86291 ] ;
alpha_c_right = [ 0.00000 ] ;
kc_right = [ -0.12643   0.03376   0.00280   -0.00400  0.00000 ] ;

R = [ 0.9996   -0.0196    0.0183 ; 0.0190    0.9994    0.0306; -0.0189   -0.0302    0.9994];
T = [ -118.50055;   -0.11324;  1.48976 ]; % If T is column vector Coefficients become column vector, if row vector 

% Stereo calibration parameters after optimization:


% Intrinsic parameters of left camera:

% Focal Length:          fc_left = [ 767.48024   764.21925 ] ? [ 1.13884   1.11887 ]
% Principal point:       cc_left = [ 314.12348   205.28206 ] ? [ 2.05972   1.95312 ]
% Skew:             alpha_c_left = [ 0.00000 ] ? [ 0.00000  ]   => angle of pixel axes = 90.00000 ? 0.00000 degrees
% Distortion:            kc_left = [ -0.12140   0.02189   0.00747   0.00224  0.00000 ] ? [ 0.00869   0.03444   0.00060   0.00082  0.00000 ]


% Intrinsic parameters of right camera:

% Focal Length:          fc_right = [ 779.92851   777.71863 ] ? [ 1.19980   1.19791 ]
% Principal point:       cc_right = [ 287.73953   188.86291 ] ? [ 2.08649   2.04651 ]
% Skew:             alpha_c_right = [ 0.00000 ] ? [ 0.00000  ]   => angle of pixel axes = 90.00000 ? 0.00000 degrees
% Distortion:            kc_right = [ -0.12643   0.03376   0.00280   -0.00400  0.00000 ] ? [ 0.00929   0.05197   0.00068   0.00069  0.00000 ]


% Extrinsic parameters (position of right camera wrt left camera):

% Rotation vector:             om = [ -0.03038   0.01857  0.01932 ] ? [ 0.00343   0.00352  0.00025 ]
% Translation vector:           T = [ -118.50055   -0.11324  1.48976 ] ? [ 0.16732   0.14749  0.92895 ]


% Note: The numerical errors are approximately three times the standard deviations (for reference).


% >> R=rodrigues(om)

% R =

    % 0.9996   -0.0196    0.0183
    % 0.0190    0.9994    0.0306
   % -0.0189   -0.0302    0.9994


 
LeftDistort = [( distortedLeftCoordinates(1) - cc_left(1) )/fc_left(1);( distortedLeftCoordinates(2) - cc_left(2) )/fc_left(2)];
RightDistort = [( distortedRightCoordinates(1) - cc_right(1) )/fc_right(1);( distortedRightCoordinates(2) - cc_right(2) )/fc_right(2)];

LeftDistort(1) = LeftDistort(1) - alpha_c_left * LeftDistort(2);
RightDistort(1) = RightDistort(1) - alpha_c_right * RightDistort(2);

k1 = kc_left(1);
k2 = kc_left(2);
k3 = kc_left(5);
p1 = kc_left(3);
p2 = kc_left(4);
    
leftTemp = LeftDistort; 				% initial guess
   
for kk=1:20,
        
    r_2 = sum(leftTemp.^2);
    k_radial =  1 + k1 * r_2 + k2 * r_2^2 + k3 * r_2^3;
    delta_x = [2*p1*leftTemp(1)*leftTemp(2) + p2*(r_2 + 2*leftTemp(1)^2);
    p1 * (r_2 + 2*leftTemp(2)^2)+2*p2*leftTemp(1)*leftTemp(2)];
    leftTemp = (LeftDistort - delta_x)./(ones(2,1)*k_radial);            
end;
LeftDistort=leftTemp;


k1 = kc_right(1);
k2 = kc_right(2);
k3 = kc_right(5);
p1 = kc_right(3);
p2 = kc_right(4);

rightTemp = RightDistort; 				% initial guess
   
for kk=1:20,
        
    r_2 = sum(rightTemp.^2);
    k_radial =  1 + k1 * r_2 + k2 * r_2^2 + k3 * r_2^3;
    delta_x = [2*p1*rightTemp(1)*rightTemp(2) + p2*(r_2 + 2*rightTemp(1)^2);
    p1 * (r_2 + 2*rightTemp(2)^2)+2*p2*rightTemp(1)*rightTemp(2)];
    rightTemp = (RightDistort - delta_x)./(ones(2,1)*k_radial);            
end;
RightDistort=rightTemp;


leftBeforeTriangulation=[LeftDistort;1];%Extending to 3D
rightBeforeTriangulation=[RightDistort;1];%%Extending to 3D

rightRotated = R'*rightBeforeTriangulation;
leftCrossRightRotated = cross(leftBeforeTriangulation,rightRotated);%leftCrossRightRotated= vector orthogonal to both leftBeforeTriangulation and rightRotated
%t=[leftBeforeTriangulation rightRotated leftCrossRightRotated]

Coefficients = inv([leftBeforeTriangulation rightRotated leftCrossRightRotated])*-T; %Coefficients=[a; b; c]
coordinatesOfThePointIn3D=( ( Coefficients(1)*leftBeforeTriangulation ) + ( -Coefficients(2)*rightRotated )-T )/2;
coordinatesOfThePointIn3D(2)=-coordinatesOfThePointIn3D(2);
coordinatesOfThePointIn3D(1)=-coordinatesOfThePointIn3D(1);

threeDcoordinates=coordinatesOfThePointIn3D;

end
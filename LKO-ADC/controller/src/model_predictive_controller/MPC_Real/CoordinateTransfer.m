function [Post_base_target, Eular_base_target] = CoordinateTransfer(world_base, world_target)
    Rworld2base = Euler2RotationFunc(world_base(4:6));
    Rworld2target = Euler2RotationFunc(world_target(4:6));
    Post_base_target= Rworld2base*(world_target(1:3)'-world_base(1:3)');
    Eular_base_target = inv(Rworld2target)*Rworld2base;
    function R = Euler2RotationFunc(euler)
        %XYZ_Euler to Rotation matrix
        x=euler(1);y=euler(2);z=euler(3);
        Rz=[cos(z),-sin(z),0;...
            sin(z),cos(z), 0;...
            0,     0     , 1];
        Ry=[cos(y), 0, sin(y);...
            0,      1,     0;...
            -sin(y),0, cos(y)];
        Rx=[1,     0,       0;...
            0, cos(x), -sin(x);....
            0, sin(x), cos(x)];
        R=(Rx*Ry*Rz)';
    end
end

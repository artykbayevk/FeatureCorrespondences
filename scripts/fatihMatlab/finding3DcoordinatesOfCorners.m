clc; clear;
load matchedCorners.mat;

matchedCornersD=double(matchedCorners);

cartesianCoordinatesOfCorners=zeros(88,100,3, 'double');
cylindricalCoordinatesOfCorners=zeros(88,100,3, 'double');

for i=1: +1: 88
    for j=1: +1: 100
        i
        j
		if(matchedCornersD(i,j,1)~=0)
			cartesianCoordinatesOfCorners(i,j,:)=findThreeDcoordinates([matchedCornersD(i,j,1) matchedCornersD(i,j,2)], [matchedCornersD(i,j,3) matchedCornersD(i,j,4)]);
            
            %1->R,2->alfa,3->z
            cylindricalCoordinatesOfCorners(i,j,1)=sqrt((cartesianCoordinatesOfCorners(i,j,3))^2+(cartesianCoordinatesOfCorners(i,j,1))^2);
            cylindricalCoordinatesOfCorners(i,j,2)=tan(cartesianCoordinatesOfCorners(i,j,1)/cartesianCoordinatesOfCorners(i,j,3));      
            cylindricalCoordinatesOfCorners(i,j,3)=cartesianCoordinatesOfCorners(i,j,2);
            
            tmp=cartesianCoordinatesOfCorners(i,j,1);%1->X,2->Y,3->Z
            cartesianCoordinatesOfCorners(i,j,1)=cartesianCoordinatesOfCorners(i,j,3);
            cartesianCoordinatesOfCorners(i,j,3)=cartesianCoordinatesOfCorners(i,j,2);
            cartesianCoordinatesOfCorners(i,j,2)=tmp;
            
        end
	end
end


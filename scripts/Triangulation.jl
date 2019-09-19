using PyCall
using Distances
using StatsBase
using LinearAlgebra
using JuMP
using Gurobi
using CSV
using Distances
using DataFrames
using SparseArrays
using Printf
using Images

leftCord = [320.0 ;220.0]
rightCord = [290.0;190.0]

fc_left = [ 767.48024 ;764.21925 ]
cc_left = [ 314.12348  ; 205.28206 ]

fc_right = [ 779.92851   ; 777.71863 ]
cc_right = [ 287.73953   ; 188.86291 ]

R = [ 0.9996   -0.0196    0.0183 ; 0.0190    0.9994    0.0306; -0.0189   -0.0302    0.9994];
T = [ -118.50055;   -0.11324;  1.48976 ];


Left_ = leftCord - cc_left
Right_ = rightCord - cc_right

LeftBeforeTr = [Left_; fc_left[1]]
RightBefireTr = [Right_; fc_right[1]]

RightRotated = R' * RightBefireTr
LeftCrossRotated = LinearAlgebra.cross(LeftBeforeTr,RightRotated)

A = [LeftBeforeTr RightRotated LeftCrossRotated]

Coef = LinearAlgebra.inv([LeftBeforeTr RightRotated  LeftCrossRotated])*-T
points = ((Coef[1]*LeftBeforeTr ) + (-Coef[2]*RightRotated) - T) /2
points[2]=-points[2]
points[1]=-points[1]

points

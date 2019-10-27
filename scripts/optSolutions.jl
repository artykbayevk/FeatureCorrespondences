using StatsBase
using LinearAlgebra
using JuMP
using Gurobi
using CSV
using Distances
using PyPlot
using SparseArrays
using Printf
using DataFrames

struct Pile
    points::Matrix{Float64} # 2 x n matrix where each column is a point
    weights::Vector{Float64}
end
Base.length(p::Pile) = length(p.weights)

function Base.rand(::Type{Pile}; n_points = rand(1:5),
                                 points = rand(2,n_points),
                                use_rand = true)
    if use_rand

        weights = rand(n_points)
#         weights = weights/sum(weights)
    else
        weights = ones(n_points)
#         weights = weights/sum(weights)
    end
    return Pile(points, weights)
end

P_dat = CSV.read("../data/artificial/P_new.csv"; header=false)
P_points = convert(Matrix, P_dat)'


Q_dat = CSV.read("../data/artificial/Q_new.csv"; header=false)
Q_points = convert(Matrix, Q_dat)'

P = rand(Pile, n_points = size(P_points)[2], points = P_points,use_rand = false)
Q = rand(Pile, n_points = size(Q_points)[2], points = Q_points, use_rand = false)

cost = pairwise(Euclidean(), P.points, Q.points; dims=2)

solCount = 1000
m = JuMP.direct_model(Gurobi.Optimizer(PoolSearchMode=2, PoolSolutions=solCount, SolutionNumber=0))

@variable(m, X[axes(cost,1), axes(cost,2)] ≥ 0, Int)
@objective(m, Min, cost ⋅ X)
@constraint(m,sum(X) .== min(sum(P.weights), sum(Q.weights)))
@constraint(m, X * ones(Int, length(Q)) .<= P.weights)
@constraint(m, X'ones(Int, length(P)) .<= Q.weights);
optimize!(m)
obj = objective_value(m)

solution_pool = zeros(solCount, length(P),length(Q))
cnt = 0
obj = objective_value(m)

for i in 0:(solCount-1)
    try
        setparam!(m.moi_backend.inner,"SolutionNumber", i)
        xn = Gurobi.get_dblattrarray(m.moi_backend.inner, "Xn", 1, length(X))
        xn_val = Gurobi.get_dblattr(m.moi_backend.inner, "PoolObjVal")
        if(round(xn_val,digits=1) != round(obj, digits=1))
            println(cnt , " solution(s) selected")
            break
        end
        default = zeros(length(P),length(Q))
        for i in 0:length(P)-1
            default[i+1,:] = xn[(i*length(Q))+1:(i+1)*length(Q)]
        end
        solution_pool[i+1,:,:] = default
        cnt+=1
    catch
        break
    end
end
sol_pool = deepcopy(solution_pool[1:cnt,:,:]);
println(size(sol_pool))

allSolutions = zeros(cnt, size(P.points)[2]*4)

for index in 1:cnt
    solOther = sparse(sol_pool[index,:,:])
    line = zeros(0)
    for (x,y,v) in zip(findnz(solOther)...)
        P_pos = [P.points[:,x][1], P.points[:,x][2]]
        Q_pos = [Q.points[:,y][1], Q.points[:,y][2]]
        append!(line, P_pos)
        append!(line, Q_pos)
    end
    allSolutions[index,:] = line
end
df = DataFrame(allSolutions)
println("Count of solutions:$(size(df)[1])")
CSV.write("../data/artificial/solutions_1.csv",  df, writeheader=false)

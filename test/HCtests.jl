using HomotopyContinuation

@var b,c,x
F = System([x^2 + b*x + c], variables=[x], parameters=[b,c])
P = randn(ComplexF64,nparameters(F))
S = solve(F; target_parameters=P)
f(b,c) = nreal(solve(F,S;start_parameters=P, target_parameters=[b,c]))
VSD = ValuedSubdivision(f; xlims=[-10, 10], ylims=[-10, 10], resolution=20, strategy=:quadtree)
refine!(VSD, 500; strategy=:quadtree) 
visualize(VSD)

#Do the above code for a general system of equations
function visualize_real_solutions(F::System)
    #Check if F has 2 parameters
    if length(parameters(F)) != 2
        throw(ArgumentError("The system must have exactly 2 parameters for visualization."))
    end
    @var x,y
    f(p1,p2) = nreal(solve(F;target_parameters=[p1,p2]))
    VSD = ValuedSubdivision(f; xlims=[-10, 10], ylims=[-10, 10], resolution=20)
    refine!(VSD, 500) 
    visualize(VSD)
    return(VSD)
end

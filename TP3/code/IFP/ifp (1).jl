print("Loading income fluctuations problem…")
using PlotlyJS, LinearAlgebra, Interpolations, Optim, Distributions

using QuantEcon: tauchen

struct IFP
	β::Float64
	γ::Float64

	r::Float64

	kgrid::Vector{Float64}
	ygrid::Vector{Float64}
	Py::Matrix{Float64}

	v::Matrix{Float64}
	gc::Matrix{Float64}
	gk::Matrix{Float64}
end
function IFP(;β=0.96, r=0.02, γ=2, Nk = 20, Ny = 25, μy = 1, ρy = 0.8, σy = 0.02)
	kgrid = range(0,1,length=Nk)

	ychain = tauchen(Ny, ρy, σy, 0, 2)
	ygrid = exp.(ychain.state_values) * μy
	Py = ychain.p

	v = zeros(Nk, Ny)
	gc = zeros(Nk, Ny)
	gk = zeros(Nk, Ny)

	return IFP(β, γ, r, kgrid, ygrid, Py, v, gc, gk)
end

u(cv, ce::IFP) = u(cv, ce.γ)
function u(cv, γ)
	if γ == 1
		return log(cv)
	else
		return cv^(1-γ) / (1-γ)
	end
end

function budget_constraint(kpv, kv, yv, r)
	c = kv * (1+r) - kpv + yv
	return c
end

function eval_value(kpv, kv, yv, py, itp_v::AbstractInterpolation, ce::IFP)
	β, r = ce.β, ce.r

	cv = budget_constraint(kpv, kv, yv, r)

	cv > 0 || return cv, -Inf
	
	ut = u(cv, ce)

	Ev = 0.0
	for (jyp, ypv) in enumerate(ce.ygrid)
		prob = py[jyp]
		Ev += prob * itp_v(kpv, ypv)
	end
	
	v = ut + β * Ev

	return cv, v
end

function opt_value(jk, jy, itp_v::AbstractInterpolation, ce::IFP)
	kv = ce.kgrid[jk]
	yv = ce.ygrid[jy]
	
	k_min = minimum(ce.kgrid)
	k_max = maximum(ce.kgrid)
	k_max = min(k_max, max(0, kv * (1+ce.r) + yv - 1e-5))

	py = ce.Py[jy,:]

	obj_f(kpv) = -eval_value(kpv, kv, yv, py, itp_v, ce)[2]

	res = Optim.optimize(obj_f, k_min, k_max, GoldenSection())

	vp = -res.minimum
	k_star = res.minimizer
	c_star = budget_constraint(k_star, kv, yv, ce.r)

	return k_star, vp, c_star
end

function vfi_iter!(new_v, itp_v::AbstractInterpolation, ce::IFP)
	for jk in eachindex(ce.kgrid), jy in eachindex(ce.ygrid)
		
		k_star, vp, c_star = opt_value(jk, jy, itp_v, ce)

		new_v[jk, jy] = vp
		ce.gc[jk, jy] = c_star
		ce.gk[jk, jy] = k_star
	end
end

function vfi!(ce::IFP; tol = 1e-8, maxiter = 2000, verbose = true)
	new_v = similar(ce.v)
	knots = (ce.kgrid, ce.ygrid)
	
	dist = 1+tol
	iter = 0

	while dist > tol && iter < maxiter
		iter += 1
		
		itp_v = interpolate(knots, ce.v, Gridded(Linear()))
		vfi_iter!(new_v, itp_v, ce)

		dist = norm(new_v - ce.v) / norm(ce.v)

		ce.v .= new_v
		verbose && print("Iteration $iter. Distance = $dist\n")
	end
	dist < tol || print("✓")
end

print(" ✓\nConstructor ce = IFP(;β=0.96, r=0.02, γ=2, Nk = 20, Ny = 25, μy = 1, ρy = 0.8, σy = 0.02)\n")
print("Solver vfi!(ce::IFP; tol = 1e-8, maxiter = 2000, verbose = true)\n")




#Simulador 

function iter_simul_ifp(kt, yt, itp_gc, itp_gk, ymin, ymax, ρ, σ)
    ct = itp_gc(kt, yt)

    kp = itp_gk(kt, yt)

    ϵt = rand(Normal(0,1))
	yp = exp(ρ * log(yt) + σ * ϵt)

    yp = max(min(ymax, yp), ymin)

    return kp, yp, ct
end

function simul(dd::IFP; k0 = mean(dd.kgrid), y0 = mean(dd.ygrid), T = 100)    
    ymin, ymax = extrema(dd.ygrid)
    ρ = 0.8
    σ = 0.02
    knots = (dd.kgrid, dd.ygrid)
    
    itp_gk = interpolate(knots, dd.gk, Gridded(Linear()))
    itp_gc = interpolate(knots, dd.gc, Gridded(Linear()))

    sample = Dict(sym => Vector{Float64}(undef, T) for sym in (:k, :y, :c))

    sample[:y][1] = y0
    sample[:k][1] = k0
    
    for jt in 1:T

        k0, y0, c = iter_simul_ifp(k0, y0, itp_gc, itp_gk, ymin, ymax, ρ, σ)

        sample[:c][jt] = c

        if jt < T
            sample[:k][jt+1] = k0
            sample[:y][jt+1] = y0
        end
    end

    return sample
end


#Gráficos del simulador
# Crear una instancia del problema IFP
ce = IFP(; β = 0.96, r = 0.02, γ = 2, Nk = 20, Ny = 25, μy = 1, ρy = 0.8, σy = 0.02)

# Resolver el problema utilizando vfi!
vfi!(ce; tol = 1e-8, maxiter = 2000, verbose = true)

# Simular el problema
simulacion = simul(ce, k0 = mean(ce.kgrid), y0 = mean(ce.ygrid), T = 10)




# Realizar análisis de la distribución ergódica de c/y y k
cy_sim = simulacion[:c] ./ simulacion[:y]
media_cy_sim = mean(cy_sim)
media_k_sim = mean(simulacion[:k])
percentil_25_cy_sim = quantile(cy_sim, 0.25)
percentil_75_cy_sim = quantile(cy_sim, 0.75)
percentil_25_k_sim = quantile(simulacion[:k], 0.25)
percentil_75_k_sim= quantile(simulacion[:k], 0.75)
media_k_sim = mean(simulacion[:k])

# Crear histograma interactivo de c/y simulado
a=histogram_cy_sim = plot(PlotlyJS.histogram(x = cy_sim, nbinsx = 10, name = "Distribución de c/y simulado"),
Layout(
	title = "Distribución de c/y simulado",
	xaxis_range = [1, 1.5],
	yaxis_range = [0, 5],
	xaxis_title = "Valor de c/y",
	yaxis_title = "Frecuencia"
))
savefig(a,"histogram_cy_sim.png")
# Crear histograma interactivo de k simulado
b=histogram_k_sim = plot(PlotlyJS.histogram(x = simulacion[:k], nbinsx = 10, name = "Distribución de k simulado"),
Layout(
	title = "Distribución de k simulado",
	xaxis_range = [0, 0.5],
	yaxis_range = [0, 5],
	xaxis_title = "Valor de k",
	yaxis_title = "Frecuencia"
)
)


combinado=plot([a, b])
combined_plot = plot([histogram_cy_sim, histogram_k_sim])
# Mostrar el gráfico combinado
 display(combined_plot)
					   # Mostrar los resultados
println("Media de c/y simulado:", media_cy_sim)
println("Media de k simulado:", media_k_sim)
println("Percentil 25 de c/y simulado:", percentil_25_cy_sim)
println("Percentil 75 de c/y simulado:", percentil_75_cy_sim)

# Mostrar los histogramas simulados
display(histogram_cy_sim)
display(histogram_k_sim)

fig = plot([histogram_cy_sim, histogram_k_sim]) 







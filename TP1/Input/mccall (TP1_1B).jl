print("McCall, J. J. 'Economics of Information and Job Search' The Quarterly Journal of Economics, 1970, vol. 84, issue 1, 113-126\n")
print("Loading codes… ")

using Distributions, LinearAlgebra, StatsBase, PlotlyJS

mutable struct McCall
	β::Float64
	γ::Float64

	b::Float64

	wgrid::Vector{Float64}
	pw::Vector{Float64}

	w_star::Float64
	v::Vector{Float64}
end

function McCall(;
	β = 0.96,
	γ = 0,
	b = 1,
	μw = 1,
	σw = 0.05,
	Nσ = 0,
	wmin = 0.5,
	wmax = 2,
	Nw = 500)

	if Nσ > 0
		wmin = μw - Nσ * σw
		wmax = μw + Nσ * σw
	end

	wgrid = range(wmin, wmax, length=Nw)

	w_star = first(wgrid)

	d = Normal(μw, σw)

	pw = [pdf(d, wv) for wv in wgrid]
	# pw = pdf.(d, wgrid)

	pw = pw / sum(pw)

	v = zeros(Nw)

	return McCall(β, γ, b, wgrid, pw, w_star, v)
end

function u(c, mc::McCall)
	γ = mc.γ

	if γ == 1
		return log(c)
	else
		return c^(1-γ) / (1-γ)
	end
end

function R(w, mc::McCall)
	## Valor de aceptar una oferta w: R(w) = u(w) + β R(w)
	β = mc.β
	return u(w, mc) / (1-β)
end

function E_v(mc::McCall)
	## Valor esperado de la función de valor integrando sobre la oferta de mañana
	Ev = 0.0
	for jwp in eachindex(mc.wgrid)
		Ev += mc.pw[jwp] * mc.v[jwp]
	end
	return Ev
end

function update_v(ac, re, EV=false)
	## Actualizar la función de valor con max(aceptar, rechazar) si EV es falso o usando la forma cerrada con el extreme value si EV es verdadero
	if EV
		χ = 0.1
		### Con Extreme Value type 1
		# Probabilidad de aceptar
		prob = exp(ac/χ)/(exp(ac/χ)+exp(re/χ))
		V = χ * log( exp(ac/χ) + exp(re/χ) )

		return V
	else
		return max(ac, re)
	end
end

function vf_iter!(new_v, mc::McCall)
    flag = 0
	## Una iteración de la ecuación de Bellman

	# El valor de rechazar la oferta es independiente del estado de hoy
	rechazar = u(mc.b, mc) + mc.β * E_v(mc)
	for (jw, wv) in enumerate(mc.wgrid)
		# El valor de aceptar la oferta sí depende de la oferta de hoy
		aceptar = R(wv, mc)

		# Para una oferta w, v(w) es lo mejor entre aceptar y rechazar
		new_v[jw] = update_v(aceptar, rechazar)

		# El salario de reserva es la primera vez que aceptar es mejor que rechazar
		if flag == 0 && aceptar >= rechazar
			mc.w_star = wv
			flag = 1
		end
	end
end

function vfi!(mc::McCall; maxiter = 2000, tol = 1e-8, verbose=true)
	dist, iter = 1+tol, 0
	new_v = similar(mc.v)
	while dist > tol && iter < maxiter
		iter += 1
		vf_iter!(new_v, mc)
		dist = norm(mc.v - new_v)
		mc.v .= new_v # volcar lo de new_v hacia mc.v, lugar a lugar
		if verbose
			print("Iter $iter: dist = $dist\n")
		end
	end
	if verbose
		if iter == maxiter
			print("Abandoné después de ")
		else
			print("Terminé en ")
		end
		print("$iter iteraciones.\nDist = $dist\n")
	end
end

function make_plots(mc::McCall)

    aceptar_todo = [R(wv, mc) for wv in mc.wgrid]
    at = scatter(x=mc.wgrid, y=aceptar_todo, line_color="#f97760", name="u(w) / (1-β)")

    rechazar_todo = [u(mc.b, mc) + mc.β * E_v(mc) for wv in mc.wgrid]
    rt = scatter(x=mc.wgrid, y=rechazar_todo, line_color="#0098e9", name="u(b) + β ∫v(z) dF(z)")

    opt = scatter(x=mc.wgrid, y=mc.v, line_color="#5aa800", line_width=3, name="v(w)")

    traces = [at, rt, opt]

    shapes = [vline(mc.w_star, line_dash="dot", line_color="#818181")]

    annotations = [attr(x=mc.w_star, y=0, yanchor="top", yref="paper", showarrow=false, text="w*")]

    layout = Layout(shapes=shapes,
        annotations=annotations,
        title="Value function in McCall's model",
        width=1920 * 0.5, height=1080 * 0.5,
        legend=attr(orientation="h", x=0.05),
        xaxis=attr(zeroline=false, gridcolor="#434343"),
        yaxis=attr(zeroline=false, gridcolor="#434343"),
        paper_bgcolor="#272929", plot_bgcolor="#272929",
        font_color="#F8F1E9", font_size=16,
        font_family="Lato",
        hovermode="x",
    )

    plot(traces, layout)
end

function simul(mc::McCall, flag = 0; maxiter = 2000, verbose::Bool=true)
	t = 0
	PESOS = Weights(mc.pw)
	while flag == 0 && t < maxiter
		t += 1
		wt = sample(mc.wgrid, PESOS)
		verbose && print("Salario en el período $t: $wt. ")
		verbose && sleep(0.1)
		wt >= mc.w_star ? flag = 1 : (verbose && println("Sigo buscando"))
	end

	(verbose && flag == 1) && println("Oferta aceptada en $t períodos")
	
	return t
end


print("✓\nConstructor mc = McCall(; β = 0.96, γ = 0, b = 1, μw = 1, σw = 0.05, wmin = 0.5, wmax = 2, Nw = 50\n")
print("Main loop vfi!(mc)\n")


#1.1 Estaticas comparadas 
import Pkg
Pkg.add("Plots")
using Plots
using PlotlyJS

# Definir valores para b y β
b_values = range(0.5, stop=1.5, length=25)
β_values = range(0.9, stop=0.99, length=25)

# Inicializar vectores para almacenar resultados
w_star_b = similar(b_values)
w_star_β = similar(β_values)

# Resolución del modelo para diferentes valores de b
for (i, b) in enumerate(b_values)
    mc = McCall(b=b)
    vfi!(mc)
    w_star_b[i] = mc.w_star
end

# Resolución del modelo para diferentes valores de β
for (i, β) in enumerate(β_values)
    mc = McCall(β=β)
    vfi!(mc)
    w_star_β[i] = mc.w_star
end

# Crear gráfico para la estática comparada de b
aceptar_todo_b = [R(wv, mc) for wv in mc.wgrid]
rechazar_todo_b = [u(mc.b, mc) + mc.β * E_v(mc) for wv in mc.wgrid]


layout1 = Layout(;title="Estatica comparada de b. (Nw=500) ",
                     xaxis=attr(title="b", showgrid=false, zeroline=false),
                     yaxis=attr(title="Función V", zeroline=false))
plot_b = plot(
    b_values, w_star_b,
    xlabel="B", ylabel="V",
    title="Efecto de cambiar b en V",
    legend=false,
	layout1
)

savefig(plot_b,"estatica_comparada_b2.png")

# Crear gráfico para la estática comparada de β
aceptar_todo_β = [R(wv, mc) for wv in mc.wgrid]
rechazar_todo_β = [u(mc.b, mc) + mc.β * E_v(mc) for wv in mc.wgrid]

layout2 = Layout(;title="Estatica comparada de β. (Nw=500)",
                     xaxis=attr(title="β", showgrid=false, zeroline=false),
                     yaxis=attr(title="Función V", zeroline=false))

plot_β = plot(
    β_values, w_star_β,
    xlabel="β", ylabel="V",
    title="Efecto de cambiar β en V",
    legend=false,
	layout2
)


savefig(plot_β,"estatica_comparada_β2.png")


#1.2 Simulaciones

using Random
using Plots


using Distributions, LinearAlgebra, StatsBase, PlotlyJS

# ... (Definición de McCall y otras funciones) ...

# Defino una función para realizar una sola simulación y devolver el tiempo de parada T
function simulate_once(mc::McCall, maxiter = 2000, verbose::Bool = false)
    return simul(mc, verbose=verbose, maxiter=maxiter)
end

# Defino una función para realizar K simulaciones y obtener un vector de tiempos de parada T
function simulate_multiple(mc::McCall, K::Int, maxiter = 2000, verbose::Bool = false)
    T = Vector{Int}(undef, K)
    for i in 1:K
        T[i] = simulate_once(mc, maxiter, verbose)
    end
    return T
end

# Paso 1: Inicializar y resolver una instancia de McCall
mc = McCall()
vfi!(mc)

# Paso 2: Elegir una cantidad K de repeticiones
K = 10000

# Paso 3: Preasignar un vector T para guardar los K valores de T
T = simulate_multiple(mc, K)

# Calcula la media y cuantiles de T
mean_T = mean(T)
quantile_25 = quantile(T, 0.25)
quantile_75 = quantile(T, 0.75)

# Paso 5: Crear un histograma con las frecuencias de los Ts
hist = histogram(T, bins=100, legend=false, title="Distribución de T. (60 bins)",
    label="T",
    xlabel="Tiempo de aceptación",
    ylabel="Frecuencia",
    linecolor=:auto,
    fillalpha=0.5, fillcolor=:blue, alpha=0.5,
    xlims=(minimum(T), maximum(T)), # Limita los límites del eje x
    lw=2, color=:green, line=:green, yticks=false, xticks=false) # Elimina las etiquetas de ejes
	
# Agregar líneas para media y cuantiles
media=vline!([mean_T], line=(:red, 2), label="Mean")
quantil25=vline!([quantile_25], line=(:orange, 2), label="Quantiles")
quantil75=vline!([quantile_75], line=(:green, 2), label="Quantiles")

savefig(hist,"histogram_with_stats.png")

# Calcula la media y cuantiles de T
mean_T = mean(T)
quantile_25 = quantile(T, 0.25)
quantile_75 = quantile(T, 0.75)

# Paso 5: Crear un histograma con las frecuencias de los Ts
hist = histogram(T, bins=80, legend=false, title="Distribución de T. (80 bins)",
    label="T",
    xlabel="Tiempo de aceptación",
    ylabel="Frecuencia",
    linecolor=:auto,
    fillalpha=0.5, fillcolor=:blue, alpha=0.5,
    xlims=(minimum(T), maximum(T)), # Limita los límites del eje x
    lw=2, color=:green, line=:green, yticks=true, xticks=true) 
# Agregar líneas para media y cuantiles con etiquetas
media=vline!([mean_T], line=(:red, 2), label="Mean")
quantil25=vline!([quantile_25], line=(:orange, 2), label="Quantile 25")
quantil75=vline!([quantile_75], line=(:green, 2), label="Quantile 75")

# Guardar el gráfico con las etiquetas
savefig(hist,"histogram_with_stats.png")




#1.3 ¿Como cambia E(T) con β?

		function calculate_expected_T(β)
			# Crear un objeto McCall con el valor de β
			mc = McCall(β=β)
			vfi!(mc)
			
			# Realizar 10,000 simulaciones y calcular la media de T
			T = simulate_multiple(mc, 10000)
			mean_T = mean(T)
			
			return mean_T
		end
		
		function calculate_expected_T_values(β_values)
			T_values = Float64[]
		
			for β in β_values
				mean_T = calculate_expected_T(β)
				push!(T_values, mean_T)
			end
			
			return T_values
		end
# Definir el rango de valores de β
β_values = range(0.9, stop=0.99, length=25)

# Calcular E[T] para cada valor de β
T_values = calculate_expected_T_values(β_values)

# Convertir el rango β_values en un vector
β_values = collect(β_values)

# Luego, crea el gráfico de dispersión
layout3 = Layout(;title="Relación entre β y E[T]. ",
xaxis=attr(title="β", showgrid=false, zeroline=false),
yaxis=attr(title="E[T]", zeroline=false),
legend=false,
width=800,  # Ancho personalizado
height=600,  # Altura personalizada
margin=attr(l=100, r=100, t=100, b=100),  # Margen personalizado
)

ej3=plot(β_values, T_values,layout3, xlabel="β", ylabel="E[T]", title="E[T] as a Function of β", legend=false)

savefig(ej3,"E[T]_as_a_Function_of_β.png")





#Robustness – opcional

print("McCall, J. J. 'Economics of Information and Job Search' The Quarterly Journal of Economics, 1970, vol. 84, issue 1, 113-126\n (ROBUST VERSION)")
print("Loading codes… ")

using Distributions, LinearAlgebra, StatsBase, PlotlyJS

mutable struct McCall
    β::Float64
    γ::Float64
    θ::Float64  # Agregue esto
    b::Float64
    wgrid::Vector{Float64}
    pw::Vector{Float64}
    w_star::Float64
    v::Vector{Float64}
end

function McCall(;
    β = 0.96,
    γ = 0,
    θ = 1.0,  # Agregue esto
    b = 1,
    μw = 1,
    σw = 0.05,
    Nσ = 0,
    wmin = 0.5,
    wmax = 2,
    Nw = 500)

    if Nσ > 0
        wmin = μw - Nσ * σw
        wmax = μw + Nσ * σw
    end

    wgrid = range(wmin, wmax, length=Nw)

    w_star = first(wgrid)

    d = Normal(μw, σw)

    pw = [pdf(d, wv) for wv in wgrid]
    # pw = pdf.(d, wgrid)

    pw = pw / sum(pw)

    v = zeros(Nw)

    return McCall(β, γ, θ, b, wgrid, pw, w_star, v)  # Añadi θ aquí
end




function u(c, mc::McCall)
	γ = mc.γ

	if γ == 1
		return log(c)
	else
		return c^(1-γ) / (1-γ)
	end
end

function R(w, mc::McCall)
	## Valor de aceptar una oferta w: R(w) = u(w) + β R(w)
	β = mc.β
	return u(w, mc) / (1-β)
end

# Definir el operador de distorsión T(X)
function T(X, θ)
    return -(1/θ) * log(mean(exp.(-θ .* X)))
end

# Modificar la función E_v(mc::McCall) para usar el operador de distorsión
function E_v(mc::McCall, θ, robust=true)
    if robust
        return T(mc.v, θ)
    else
        return mean(mc.v)
    end
end

# Modificar la ecuación de Bellman en la función vf_iter!(new_v, mc::McCall)
function vf_iter!(new_v, mc::McCall, θ, robust=true)
    rechazar = u(mc.b, mc) + mc.β * E_v(mc, θ, robust)
    for (jw, wv) in enumerate(mc.wgrid)
        aceptar = R(wv, mc)
        new_v[jw] = max(aceptar, rechazar)
        if aceptar >= rechazar
            mc.w_star = wv
            break
        end
    end
end

# Modificar la función vfi!(mc::McCall; maxiter = 2000, tol = 1e-8) para incluir θ y robust como argumentos
function vfi!(mc::McCall, θ, robust=true; maxiter = 2000, tol = 1e-8)
    dist, iter = 1+tol, 0
    new_v = similar(mc.v)
    while dist > tol && iter < maxiter
        iter += 1
        vf_iter!(new_v, mc, θ, robust)
        dist = norm(mc.v - new_v)
        mc.v .= new_v
    end
end

# Definir rangos para θ y w⋆
θ_range = range(0.1, stop=1.0, length=100)  # Limita el rango de θ aquí

w_star_range = range(0.1, stop=1.0, length=100)
# Inicializar arreglos para almacenar los resultados
E_T_values = []
w_star_values = []

# Calcular E[T] y w⋆ para cada valor de θ 

#ACÁ SALTA EL ERROR
for θ in θ_range
    mc = McCall(β=0.96, γ=0.0, b=1.0, μw=1.0, σw=0.05, Nσ=0.0,wmin=0.5,wmax=2.0,Nw=500 ,θ=θ) 
    vfi!(mc; θ=θ, robust=true) # Acà tiene que ir un true
    push!(E_T_values, E_v(mc; θ=θ, robust=true)) # Modifica esta línea
    push!(w_star_values, mc.w_star)
end

# Crear gráficos
p1 = plot(θ_range,E_T_values,label="E[T]",xlabel="θ",ylabel="E[T]",title="E[T] vs θ")
p2 = plot(θ_range,w_star_values,label="w⋆",xlabel="θ",ylabel="w⋆",title="w⋆ vs θ")

# Mostrar gráficos
plot(p1,p2)



print("McCall, J. J. 'Economics of Information and Job Search' The Quarterly Journal of Economics, 1970, vol. 84, issue 1, 113-126\n (ROBUST VERSION)")
print("Loading codes… ")

using Distributions, LinearAlgebra, StatsBase, PlotlyJS, Plots

mutable struct McCall
    β::Float64
    γ::Float64
    θ::Float64  # Agregue esto
    b::Float64
    wgrid::Vector{Float64}
    pw::Vector{Float64}
    w_star::Float64
    v::Vector{Float64}
end

function McCall(;
    β = 0.96,
    γ = 0,
    θ = 1.0,  # Agregue esto
    b = 1,
    μw = 1,
    σw = 0.05,
    Nσ = 0,
    wmin = 0.5,
    wmax = 2,
    Nw = 500)

    if Nσ > 0
        wmin = μw - Nσ * σw
        wmax = μw + Nσ * σw
    end

    wgrid = range(wmin, wmax, length=Nw)

    w_star = first(wgrid)

    d = Normal(μw, σw)

    pw = [pdf(d, wv) for wv in wgrid]
    # pw = pdf.(d, wgrid)

    pw = pw / sum(pw)

    v = zeros(Nw)

    return McCall(β, γ, θ, b, wgrid, pw, w_star, v)  # Añadi θ aquí
end

function u(c, mc::McCall)
    γ = mc.γ

    if γ == 1
        return log(c)
    else
        return c^(1-γ) / (1-γ)
    end
end

function R(w, mc::McCall)
    β = mc.β
    return u(w, mc) / (1-β)
end

# Definir el operador de distorsión T(X)
function T(X, θ)
    return -(1/θ) * log(mean(exp.(-θ .* X)))
end

# Modificar la función E_v(mc::McCall) para usar el operador de distorsión
function E_v(mc::McCall, θ, robust=true)
    if robust
        X = max.(exp.(-θ .* mc.v), eps())
        return T(X, θ)
    else
        return mean(mc.v)
    end
end
# Modificar la ecuación de Bellman en la función vf_iter!(new_v, mc::McCall)
function vf_iter!(new_v, mc::McCall, θ, robust=true)
    rechazar = u(mc.b, mc) + mc.β * E_v(mc, θ, robust)
    for (jw, wv) in enumerate(mc.wgrid)
        aceptar = R(wv, mc)
        new_v[jw] = max(aceptar, rechazar)
        if aceptar >= rechazar
            mc.w_star = wv
            break
        end
    end
end

# Modificar la función vfi!(mc::McCall; maxiter = 2000, tol = 1e-8) para incluir θ y robust como argumentos
function vfi!(mc::McCall, θ, robust=true; maxiter = 2000, tol = 1e-8)
    dist, iter = 1+tol, 0
    new_v = similar(mc.v)
    while dist > tol && iter < maxiter
        iter += 1
        vf_iter!(new_v, mc, θ, robust)
        dist = norm(mc.v - new_v)
        mc.v .= new_v
    end
end

# Definir rangos para θ y w⋆
θ_range = range(0.1, stop=2.0, length=100)  # Limita el rango de θ aquí

w_star_range = range(0.1, stop=.0, length=100)
# Inicializar arreglos para almacenar los resultados
E_T_values = []
w_star_values = []

# Calcular E[T] y w⋆ para cada valor de θ 

#ACÁ SALTA EL ERROR
for θ in θ_range
    mc = McCall(β=0.96, γ=0.0, b=1.0, μw=1.0, σw=0.05, Nσ=0.0,wmin=0.5,wmax=2.0,Nw=500 ,θ=θ) 
    vfi!(mc, θ, true) # Acà tiene que ir un true
    push!(E_T_values, E_v(mc, θ, true)) # Modifica esta línea
    push!(w_star_values, mc.w_star)
end

# Crear gráficos
using Pkg
Pkg.add("PlotlyJS")
Pkg.add("Plots")
Pkg.add("GR")
using Plots
# Crear gráfico de E[T] vs θ
plot1 =Plots.plot(θ_range, E_T_values, label="E[T]", xlabel="θ", ylabel="E[T]", title="E[T] vs θ",legend=false)

# Crear gráfico de w⋆ vs θ
plot2 = Plots.plot(θ_range, w_star_values, label="w⋆", xlabel="θ", ylabel="w⋆", title="w⋆ vs θ")

# Mostrar los gráficos
Plots.plot(plot1, plot2, layout=(1,2), legend=false)


Pkg.add("PyPlot")
using PyPlot
# Crear gráfico de E[T] vs θ
figure()
plot(θ_range, E_T_values, label="E[T]")
xlabel("θ")
ylabel("E[T]")
title("E[T] vs θ")

# Crear gráfico de w⋆ vs θ
figure()
plot(θ_range, w_star_values, label="w⋆")
xlabel("θ")
ylabel("w⋆")
title("w⋆ vs θ")

show()
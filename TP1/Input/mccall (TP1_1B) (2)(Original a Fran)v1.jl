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


using Distributions, LinearAlgebra, StatsBase, PlotlyJS, Random


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
hist = histogram(T, bins=80, legend=false, title="Distribución de T. (80 bins)",
    label="T",
    xlabel="Tiempo de aceptación",
    ylabel="Frecuencia",
    linecolor=:auto,
    fillalpha=0.5, fillcolor=:blue, alpha=0.5,
    xlims=(minimum(T), maximum(T)), # Limita los límites del eje x
    lw=2, color=:green, line=:green, yticks=true, xticks=true) 
# Agregar líneas para media y cuantiles 
media=vline!([mean_T], line=(:red, 2), label="Mean")
quantil25=vline!([quantile_25], line=(:orange, 2))
quantil75=vline!([quantile_75], line=(:green, 2))

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
    θ = 1,  # Agregue esto
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



# Definir el operador de distorsión Operador(X, θ)
#Metodo alternativo

function E_v(mc::McCall;robust=true)
	Ev=0.0
	if robust
		for jwp in eachindex(mc.wgrid)
			Ev2 += exp((-mc.θ)*(mc.v[jwp]))*mc.pw[jwp]
		end
		Ev = (-(1/mc.θ)) * log(Ev2)
	return Ev
	else
		for jwp in eachindex(mc.wgrid)
			Ev += mc.pw[jwp] * mc.v[jwp]
		end
		return Ev
	end
end

# Modificar la función E_v(mc::McCall) para usar el operador de distorsión
#Metodo alternativo
function E_v(mc::McCall; θ=θ, robust=true)
    if robust
        return Operador(mc, θ)
    else
        # Valor esperado de la función de valor integrando sobre la oferta de mañana
        Ev = 0.0
        for jwp in eachindex(mc.wgrid)
            Ev += mc.pw[jwp] * mc.v[jwp]
        end
        return Ev
    end
end

function robustness(;K=600,  θmin=0.5, θmax=1)
	θgrid2 = range(θmin, θmax, K)
	Tgrid = similar(θgrid2)
	for (jθ, θp) in enumerate(θgrid2)
		Tgrid[jθ]=dist(;θ2=θp)
	end
end

# Modificar la ecuación de Bellman en la función vf_iter!(new_v, mc::McCall)
#Metodo alternativo
function vf_iter!(new_v, mc::McCall; θ=θ, robust=true)
    rechazar = u(mc.b, mc) + mc.β * Operador(mc, θ)
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
#Metodo alternativo
function vfi!(mc::McCall; θ=θ, robust=true, maxiter=10000, tol=1e-8)
    dist, iter = 1.0 + tol, 0
    new_v = similar(mc.v)
    while dist > tol && iter < maxiter
        iter += 1
        vf_iter!(new_v, mc; θ=θ, robust=robust)  # Pasa θ correctamente aquí
        dist = norm(mc.v - new_v)
        copyto!(mc.v, new_v)
    end
end

# Definir rangos para θ y w⋆
θ_range = range(0.1, stop=1.0, length=100)  # Limita el rango de θ aquí

w_star_range = range(0.1, stop=1.0, length=100)
# Inicializar arreglos para almacenar los resultados
E_T_values = vec(zeros(length(θ_range)))
w_star_values = vec(zeros(length(w_star_range)))

# Calcular E[T] y w⋆ para cada valor de θ 

for (iθ, jθ) in enumerate(θ_range)
    mc = McCall(β=0.96, γ=0.0, b=1.0, μw=1.0, σw=0.05, Nσ=0.0,wmin=0.5,wmax=2.0,Nw=500 ,θ=jθ) 
    vfi!(mc; θ = jθ, robust = true) # Acà tiene que ir un true
    #push!(E_T_values, E_v(mc; θ = θ, robust = true)) # Modifica esta línea
    w_star_values[iθ] = mc.w_star
	
	robustness_values[iθ] = robustness(;θ2=jθ)
end



# Crear gráficos
using Plots 
layout5 = Layout(;title="Relación entre θ y E[T]. ",
xaxis=attr(title="θ", showgrid=false, zeroline=false),
yaxis=attr(title="E[T]", zeroline=false),
legend=false,
width=800,  # Ancho personalizado
height=600,  # Altura personalizada
margin=attr(l=100, r=100, t=100, b=100),  # Margen personalizado
)

p1 = plot(θ_range,E_T_values,label="E[T]",xlabel="θ",ylabel="E[T]",title="E[T] vs θ",layout4)
p2 = plot(θ_range,w_star_values,label="w⋆",xlabel="θ",ylabel="w⋆",title="w⋆ vs θ")

# Mostrar gráficos
plot(p1,p2)



function dist(; θ2=0.0)
    # Realiza cálculos para la función dist y retorna el resultado
    # Reemplaza esto con los cálculos reales que necesitas realizar
    return θ2 * 2.0
end

# Función para calcular robustness
function robustness(; K=600, θmin=0.5, θmax=1)
    θgrid2 = range(θmin, θmax, K)
    Tgrid = similar(θgrid2)

    for (jθ, θp) in enumerate(θgrid2)
        Tgrid[jθ] = dist(θ2=θp)  # Llama a la función dist con θ2=θp
    end

    return Tgrid  # Devuelve el arreglo Tgrid o el valor necesario
end
θ_range = range(0.1, stop=1.0, length=100)  # Limita el rango de θ aquí

w_star_range = range(0.1, stop=1.0, length=100)
# Inicializar arreglos para almacenar los resultados
E_T_values = vec(zeros(length(θ_range)))
w_star_values = vec(zeros(length(w_star_range)))

# Calcular E[T], w⋆ y robustness para cada valor de θ
for (iθ, jθ) in enumerate(θ_range)
    mc = McCall(β=0.96, γ=0.0, b=1.0, μw=1.0, σw=0.05, Nσ=0.0, wmin=0.5, wmax=2.0, Nw=500, θ=jθ)
    vfi!(mc; θ = jθ, robust = true)
    E_T_values[iθ] = E_v(mc; θ = jθ, robust = true)
    w_star_values[iθ] = mc.w_star
    robustness_values[iθ] = robustness(θ2=jθ)  # Calcula el valor de robustness

end

p3 = plot(θ_range, robustness, label="Robustness", xlabel="θ", ylabel="Robustness", title="Robustness vs θ")



plot(robustness,layout5)








using Distributions, LinearAlgebra, PlotlyJS

mutable struct McCall
    β::Float64
    γ::Float64
    θ::Float64
    b::Float64
    wgrid::Vector{Float64}
    pw::Vector{Float64}
    w_star::Float64
    v::Vector{Float64}
end

function McCall(;
    β = 0.96,
    γ = 0.0,
    θ = 1.0,
    b = 1.0,
    μw = 1.0,
    σw = 0.05,
    Nσ = 0.0,
    wmin = 0.5,
    wmax = 2.0,
    Nw = 500
)

    if Nσ > 0
        wmin = μw - Nσ * σw
        wmax = μw + Nσ * σw
    end

    wgrid = range(wmin, wmax, length=Nw)
    w_star = first(wgrid)
    d = Normal(μw, σw)
    pw = [pdf(d, wv) for wv in wgrid]
    pw = pw / sum(pw)
    v = zeros(Nw)

    return McCall(β, γ, θ, b, wgrid, pw, w_star, v)
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

function E_v(mc::McCall; robust=true)
    Ev = 0.0
    if robust
        Ev2 = 0.0
        for jwp in eachindex(mc.wgrid)
            Ev2 += exp((-mc.θ) * mc.v[jwp]) * mc.pw[jwp]
        end
        Ev = (-(1/mc.θ)) * log(Ev2)
    else
        for jwp in eachindex(mc.wgrid)
            Ev += mc.pw[jwp] * mc.v[jwp]
        end
    end
    return Ev
end


function vf_iter!(new_v, mc::McCall; θ=θ, robust=true)
    rechazar = u(mc.b, mc) + mc.β * Ev(mc, θ)
    for (jw, wv) in enumerate(mc.wgrid)
        aceptar = R(wv, mc)
        new_v[jw] = max(aceptar, rechazar)
        if aceptar >= rechazar
            mc.w_star = wv
            break
        end
    end
end

function vfi!(mc::McCall; θ=θ, robust=true, maxiter=10000, tol=1e-8)
    dist, iter = 1.0 + tol, 0
    new_v = similar(mc.v)
    while dist > tol && iter < maxiter
        iter += 1
        vf_iter!(new_v, mc; θ=θ, robust=robust)
        dist = norm(mc.v - new_v)
        copyto!(mc.v, new_v)
    end
end

function robustness(mc::McCall, K=600, θmin=0.5, θmax=1)
    θgrid2 = range(θmin, θmax, K)
    Tgrid = similar(θgrid2)
    for (jθ, θp) in enumerate(θgrid2)
        Tgrid[jθ] = E_v(mc, θp)
    end
    return Tgrid
end

# Definir rangos para θ
θ_range = range(0.1, stop=1.0, length=100)

# Inicializar arreglos para almacenar los resultados
E_T_values = vec(zeros(length(θ_range)))
w_star_values = vec(zeros(length(θ_range)))
robustness_values = vec(zeros(length(θ_range)))

# Calcular E[T], w⋆ y robustness para cada valor de θ
for (iθ, jθ) in enumerate(θ_range)
    mc = McCall(β=0.96, γ=0.0, b=1.0, μw=1.0, σw=0.05, Nσ=0.0, wmin=0.5, wmax=2.0, Nw=500, θ=jθ)
    vfi!(mc; θ=jθ, robust=true)
    E_T_values[iθ] = E_v(mc, robust=true)
    w_star_values[iθ] = mc.w_star
    robustness_values[iθ] = Operador(mc, jθ)
end

# Crear gráficos con PlotlyJS
p1 = plot(θ_range, E_T_values, label="E[T]", xlabel="θ", ylabel="E[T]", title="E[T] vs θ")
p2 = plot(θ_range, w_star_values, label="w⋆", xlabel="θ", ylabel="w⋆", title="w⋆ vs θ")
p3 = plot(θ_range, robustness_values, label="Robustness", xlabel="θ", ylabel="Robustness", title="Robustness vs θ")

# Combina los gráficos en un diseño de gráficos (subplot)
layout = vcat(p1, p2, p3)

# Muestra el gráfico
display(layout)

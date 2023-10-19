#Ejercicio 1 

print("A cake-eating problem\nLoading codes…")

using PlotlyJS, LinearAlgebra

# Para forma cerrada
function seq(n, β, γ, k)
	ρ = β^(1/γ)
	return [k * (1-ρ) * ρ^t for t in 0:n]
end

function make_plots(; N = 250, β = 0.96, γ = 2, k = 1)

	ys = seq(N, β, γ, k)

	sc = [
		bar(x=0:N, y=ys, name = "<i>c<sub>t", marker_line_width = 0)
		scatter(x=0:N, y=cumsum(ys), yaxis="y2", name="consumo total")
		]
	
	layout = Layout(
		title = "Problema de la torta (<i>β</i> = $β, <i>γ</i> = $γ, <i>k</i> = $k)",
		width = 1920*0.5, height = 1080*0.5,
		legend = attr(orientation = "h", x = 0.05),
		xaxis = attr(zeroline = false, gridcolor="#434343"),
		yaxis1 = attr(range = [0, ys[1]*1.05], zeroline = false, gridcolor="#434343"),
		yaxis2 = attr(overlaying="y1", side="right", range=[0,k*1.061], zeroline = false, gridcolor="#434343", tick0 = 0, dtick = 0.25), 
		paper_bgcolor="#272929", plot_bgcolor="#272929",
		font_color="#F8F1E9", font_size = 18, 
)

	return plot(sc, layout)
end

# savefig(p1, "Images/cake_1_dark.pdf", height = 500, width = 1000)

# Para resolver
struct CakeEating
	β::Float64
	γ::Float64

	r::Float64

	kgrid::Vector{Float64}

	v::Vector{Float64}
	gc::Vector{Float64}
	gk::Vector{Float64}
end


function CakeEating(;β=0.96, k0 = 1, r=0.02, γ=2, Nk = 20)
	kgrid = range(1e-6,k0,length=Nk)

	v = zeros(Nk)
	gc = zeros(Nk)
	gk = zeros(Nk)

	return CakeEating(β, γ, r, kgrid, v, gc, gk)
end

# Dos métodos pasando la aversión al riesgo o el modelo entero
u(cv, ced::CakeEating) = u(cv, ced.γ)
function u(cv, γ::Number)
	if γ == 1
		return log(cv)
	else
		return cv^(1-γ) / (1-γ)
	end
end

function budget_constraint(kpv, kv, r)
	# La torta de hoy (más intereses) financia el consumo de hoy y la torta de mañana
	c = kv * (1+r) - kpv
	return c
end

function eval_value(jkp, kv, ce::CakeEating)
	# Evalúa la función de valor en estado k cuando el candidado de mañana es el jkp-ésimo
	β, r = ce.β, ce.r
	kpv = ce.kgrid[jkp]

	# Consumo implicado por la restricción de presupuesto
	cv = budget_constraint(kpv, kv, r)

	# Devuelve -Inf si el consumo llega a no ser positivo
	if cv > 0
	else
		return cv, -Inf
	end
	# cv > 0 || return cv, -Inf
	
	# Flujo de utilidad por el consumo de hoy
	ut = u(cv, ce)
	
	# Valor de continuación
	vp = ce.v[jkp]

	# Flujo más valor descontado de mañana
	v = ut + β * vp

	return cv, v
end

function opt_value(jk, ce::CakeEating)
	# Elige la torta de mañana en el jk-ésimo estado
	kv = ce.kgrid[jk]
	
	vp = -Inf
	c_star = 0.0
	k_star = 0.0
	problema = true
	# Recorre todos los posibles valores de la torta de mañana
	for (jkp, kpv) in enumerate(ce.kgrid)
		# Consumo y valor de ahorrar kpv
		cv, v = eval_value(jkp, kv, ce)

		# Si el consumo es positivo y el valor es más alto que el más alto que encontré hasta ahora, reemplazo
		if cv > 0 && v > vp
			# Actualizo el "mejor valor" con el que acabo de calcular
			vp = v
			k_star = kpv
			problema = false
			c_star = cv
		end
	end

	# Si en cualquier momento entré en el "if" de arriba, va a ser falso que hubo problemas, si no...
	if problema == true
		throw(error("Hay problemas!!"))
	end

	return k_star, vp, c_star
end

function vfi_iter!(new_v, ce::CakeEating)
	# Una iteración de la ecuación de Bellman
	# Recorro los estados posibles para hoy
	for jk in eachindex(ce.kgrid)
		
		# Calculo el mejor ahorro, consumo, y valor en el estado kv
		k_star, vp, c_star = opt_value(jk, ce)
		
		# Guardo en new_v y las funciones de política directamente
		new_v[jk] = vp
		ce.gc[jk] = c_star
		ce.gk[jk] = k_star
	end
end

function vfi!(ce::CakeEating; tol = 1e-8, maxiter = 2000, verbose = true)
	# Itera hasta convergencia de la ecuación de Bellman
	# Preparo un vector para guardar la actualización del guess
	new_v = similar(ce.v)

	dist = 1+tol
	iter = 0

	while dist > tol && iter < maxiter
		iter += 1

		# Una iteración
		vfi_iter!(new_v, ce)

		# Distancia entre el guess viejo y el nuevo
		dist = norm(new_v - ce.v) / max(1,norm(ce.v))

		# Actualizo cada elemento del guess con la nueva función de valor
		ce.v .= new_v
	end
	verbose && print("Iteration $iter. Distance = $dist\n")
	dist < tol && print("✓")
end

print(" ✓\nConstructor: CakeEating(;β=0.96, r=0.02, γ=2, Nk = 20)\nSolver: vfi!(ce::CakeEating)\n")




# Crear un objeto CakeEating con parámetros por defecto
ce = CakeEating(;β=0.96, k0 = 1, r=0.02, γ=2, Nk = 2000)

#1 Vamos con el método sencillo
#Resolver el problema utilizando vfi!
vfi!(ce)

ck = ce.gc ./ ce.kgrid
scatter1b = scatter(x=ce.kgrid, y=ck)
p = plot(scatter1b)

ce_default=CakeEating(;β=0.96, k0 = 1, r=0.02, γ=2, Nk = 2000)

vfi!(ce_default)

#Creo la función para el consumo
c_solo = ce_default.gc
# Crear un rango de valores de capital
k_values = ce_default.kgrid
# Calcular la función de consumo como función del capital (c/k)
c_over_k = ce_default.gc ./ k_values

#Calculo el ahorro 
savs=ce_default.gk


# Calcular la función de ahorro como función del capital (k'/k)
savs_prime_over_k = ce_default.gk ./ k_values

#2 Crear un gráfico de la función de consumo c/k como función de k
plot_consumo = plot(
    scatter(x=k_values, y=c_solo, name="Consumo (c)"),
    Layout(xaxis_title="Capital (k)", yaxis_title="Consumo (c)", title="Función de Consumo (c) vs Capital")
)

#3 Crear un gráfico de la función de consumo c/k como función de k
plot_c_over_k = plot(
    scatter(x=k_values, y=c_over_k, name="Consumo / Capital (c/k)"),
    Layout(xaxis_title="Capital (k)", yaxis_title="Consumo / Capital (c/k)", title="Función de Consumo (c/k) vs Capital")
)
#4 Crear un gráfico de la función de ahorro k'/k como función de k

plot_s_over_k = plot(
    scatter(x=k_values, y=savs_prime_over_k,name="Ahorro / Capital (k'/k)"),
    Layout(xaxis_title="Capital (k)", yaxis_title="Ahorro / Capital (k'/k)", title="Función de Ahorro (k'/k) vs Capital")
)



a=[plot_consumo plot_c_over_k
plot_s_over_k]

savefig(a,"vfi.png", height = 700, width = 1500)



#Voy a hacerlo con interpolaciones

#Voy a definir primero otro elemento ce así no me piso.
ce_defaultitp = CakeEating(;β=0.96, k0 = 1, r=0.02, γ=2, Nk = 2000)
# Resulvo el problema usando vfi_itp!
vfi_itp!(ce_defaultitp)

# Crear un rango de valores de capital
k_values_itp = ce_defaultitp.kgrid

# Calcular la función de consumo como función del capital (c/k)
c_over_k_itp = ce_defaultitp.gc ./ k_values_itp
#Calculo la función solo del consumo 
c_solo_itp = ce_defaultitp.gc

# Calcular la función de ahorro como función del capital (k'/k)
savs_prime_over_k_itp = ce_defaultitp.gk ./ k_values_itp

#5 Crear un gráfico de la función de consumo c/k como función de k


plot_c_over_k_itp = plot(
	scatter(x=k_values_itp, y=c_over_k_itp, name="Consumo / Capital (c/k)"),
	Layout(xaxis_title="Capital (k)", yaxis_title="Consumo / Capital (c/k)", title="Función de Consumo (c/k) vs Capital")
)

#6 Crear un gráfico de la función de ahorro k'/k como función de k

plot_s_over_k_itp = plot(
	scatter(x=k_values_itp, y=savs_prime_over_k_itp, name="Ahorro/Capital (k'/k)" ),
	Layout(xaxis_title="Capital (k)", yaxis_title="Ahorro / Capital (k'/k)", title="Función de Ahorro (k'/k) vs Capital")
)

#7 Crear un gráfico de la función de consumo c como función de k
c_plot_itp = plot(
	scatter(x=k_values_itp, y=c_solo_itp,name="Consumo (c)"),Layout(xaxis_title="Capital (k)", yaxis_title="Consumo (c)", title="Función de Consumo (c) vs Capital")
)


b = [c_plot_itp plot_c_over_k_itp
 plot_s_over_k_itp]


savefig(b, "vfiitp.png", height = 700, width = 1500)






 ### Ejercicio 2 - Simulador de torta

#1. Establezco un tiempo máximo y un capital inicial
ce_resuelto=CakeEating(;β=0.96, k0 = 1, r=0.02, γ=2, Nk = 2000)
vfi_itp!(ce_resuelto)

function SimuladorTorta(;T=300, k0=1, ce=ce_resuelto)
    #2. Creo un vector de capital y consumo
    k = zeros(T)
    c = zeros(T)
    #3. Establezco el capital inicial
    k[1] = k0

    #Establezco los interpoladores
    knt=(ce_resuelto.kgrid,)
    gk=(ce_resuelto.gk)
    gc=(ce_resuelto.gc)


    gk_interpolate= interpolate(knt, gk, Gridded(Linear()))
    gc_interpolate = interpolate(knt, gc, Gridded(Linear()))

   # 5. Completar los vectores de consumo y capital
   for t in 1:T
    c[t] = gc_interpolate([k[t]])[1]
    if t < T
        k[t + 1] = gk_interpolate([k[t]])[1]
    end
end

return k, c
end

# Llamamos a la función SimuladorTorta
k_simulado, c_simulado = SimuladorTorta()

# Imprime los resultados
function imprimir(;T=300,k0=1,ce=ce_resuelto)
    
    for t in 1:T
    println("t = $t: k = $(k_simulado[t]), c = $(c_simulado[t])")
    end
end

imprimir()



#2. Creo una función que grafique el capital y el consumo
function plot_simulacion(k_simulado, c_simulado)
    # Creo un vector de tiempo
    t = 1:length(k_simulado)

    # Creo un gráfico de capital y consumo
    plot_k = plot(
        scatter(x=t, y=k_simulado, name="Capital (k)"),
        Layout(xaxis_title="Tiempo (t)", yaxis_title="Capital (k)", title="Capital vs Tiempo")
    )
    plot_c = plot(
        scatter(x=t, y=c_simulado, name="Consumo (c)"),
        Layout(xaxis_title="Tiempo (t)", yaxis_title="Consumo (c)", title="Consumo vs Tiempo")
    )

    # Devuelvo el gráfico
    return plot_k, plot_c
end

c=plot_simulacion(k_simulado, c_simulado)

d = [c[1] c[2]]	

savefig(d,"varvst.png", height = 700, width = 1500)


function area_chart_with_simulador(simulador)
    k_simulado, c_simulado = simulador()
    T = length(k_simulado)
    t = 1:T
    torta_restante = k_simulado - c_simulado

    trace1 = scatter(x=t, y=c_simulado, fill="tozeroy", mode="none", name="Consumo (c)")
    trace2 = scatter(x=t, y=torta_restante, fill="tonexty", mode="none", name="Torta Restante (k - c)")

    plot([trace1, trace2], Layout(title="Consumo y Torta Restante", xaxis_title="Tiempo (t)", yaxis_title="Consumo y Capital", legend=:topright))
end

torta = area_chart_with_simulador(SimuladorTorta)


# Define la función para crear el gráfico de flujo acumulado
function area_chart_with_cumulative_consumption(simulador)
    k_simulado, c_simulado = simulador()
    T = length(k_simulado)
    t = 1:T
    torta_restante = k_simulado - c_simulado
    cumulative_consumption = cumsum(c_simulado)

    trace1 = scatter(x=t, y=cumulative_consumption, mode="lines", name="Flujo Acumulado de Consumo")
    trace2 = scatter(x=t, y=torta_restante, fill="tozeroy", mode="none", name="Torta Restante (k - c)")

    plot([trace1, trace2], Layout(title="Consumo y Torta Restante", xaxis_title="Tiempo (t)", yaxis_title="Consumo acumulado y Capital", legend=:topright))
end

# Llama a la función para crear el gráfico de flujo acumulado
acumulado_plots = area_chart_with_cumulative_consumption(SimuladorTorta)

p2 = area_chart_with_cumulative_consumption(SimuladorTorta)
p2= plot(acumulado_plots)

p=[ torta p2;]
p

savefig(p, "torta.png", height = 700, width = 1500)



plot([p1, p2, p3, p4,],
    Layout(grid=attr(rows=2, columns=2)

    )
)
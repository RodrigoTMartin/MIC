#Ejercicio 1 
# Crear un objeto CakeEating con parámetros por defecto
ce_default = CakeEating()

#1 Resolver el problema utilizando vfi!
vfi!(ce_default)

#Creo la función para el consumo
c_solo = ce_default.gc
# Crear un rango de valores de capital
k_values = ce_default.kgrid

# Calcular la función de consumo como función del capital (c/k)
c_over_k = ce_default.gc ./ k_values

# Calcular la función de ahorro como función del capital (k'/k)
savs_prime_over_k = ce_default.gk ./ k_values

#2 Crear un gráfico de la función de consumo c/k como función de k
plot_consumo = plot(
    scatter(x=k_values, y=c_solo),
    Layout(xaxis_title="Capital (k)", yaxis_title="Consumo (c)", title="Función de Consumo c vs Capital")
)

#3 Crear un gráfico de la función de consumo c/k como función de k
plot_c_over_k = plot(
    scatter(x=k_values, y=c_over_k, name="Consumo / Capital (c/k)"),
    Layout(xaxis_title="Capital (k)", yaxis_title="Consumo / Capital (c/k)", title="Función de Consumo c/k vs Capital")
)
#4 Crear un gráfico de la función de ahorro k'/k como función de k

plot_c_over_k = plot(
    scatter(x=k_values, y=savs_prime_over_k),
    Layout(xaxis_title="Capital (k)", yaxis_title="Ahorro / Capital (k'/k)", title="Función de Ahorro k'/k vs Capital")
)

#Voy a hacerlo con interpolaciones

#Voy a definir primero otro elemento ce así no me piso.
ce_defaultitp = CakeEating()
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
	Layout(xaxis_title="Capital (k)", yaxis_title="Consumo / Capital (c/k)", title="Función de Consumo c/k vs Capital")
)

#6 Crear un gráfico de la función de ahorro k'/k como función de k

plot_k_over_k_itp = plot(
	scatter(x=k_values_itp, y=savs_prime_over_k_itp),
	Layout(xaxis_title="Capital (k)", yaxis_title="Ahorro / Capital (k'/k)", title="Función de Ahorro k'/k vs Capital")
)

#7 Crear un gráfico de la función de consumo c como función de k
c_plot_itp = plot(
	scatter(x=k_values_itp, y=c_solo_itp),Layout(xaxis_title="Capital (k)", yaxis_title="Consumo (c)", title="Función de Consumo c vs Capital")
)




### Ejercicio 2 - Simulador de torta

#1. Establezco un tiempo máximo y un capital inicial



using Interpolation, PlotlyJS

# Función para simular el problema de la torta y mostrar gráficos con PlotlyJS
function simular_torta_plotlyjs(ce, T, k0)
    # Inicializar vectores para guardar las sucesiones de consumo y capital
    C = zeros(T+1)
    K = zeros(T+1)

    # Establecer el estado inicial
    C[1] = k0
    K[1] = k0

    # Crear interpoladores para las funciones de consumo y ahorro
    interpolate_consumo(k) = interpolate(ce.kgrid, ce.gc,Gridded(Linnear()))(k)
    interpolate_ahorro(k) = interpolate(ce.kgrid, ce.gk,Gridded(Linear()))(k)

    # Simular el comportamiento a lo largo del tiempo
    for t in 1:T
        k = K[t]

        # Usar el interpolador para obtener el consumo en el estado actual
        C[t] = interpolate_consumo(k)

        # Usar el interpolador para obtener el capital en el próximo período
        K[t+1] = interpolate_ahorro(k)
    end

    # Crear un gráfico interactivo del consumo a lo largo del tiempo como flujo
    plot_consumo_flujo = scatter(
        x=0:T, y=C, mode="lines+markers", name="Consumo", xaxis="x1", yaxis="y1"
    )

    # Crear un gráfico interactivo de la fracción del consumo con respecto a la torta que queda
    fraccion_consumo = C ./ (ce.kgrid[1] .- K)
    plot_fraccion_consumo = scatter(
        x=0:T, y=fraccion_consumo, mode="lines+markers", name="Fracción de Consumo", xaxis="x2", yaxis="y2"
    )

    # Crear un gráfico interactivo del consumo acumulado a lo largo del tiempo
    flujo_acumulado_consumo = cumsum(C)
    plot_consumo_acumulado = scatter(
        x=0:T, y=flujo_acumulado_consumo, mode="lines+markers", name="Consumo Acumulado", xaxis="x3", yaxis="y3"
    )

    # Crear un layout para subplots
    layout = Layout(
        grid=attr(rows=3, columns=1),
        xaxis1=attr(title="Tiempo"),
        yaxis1=attr(title="Consumo"),
        xaxis2=attr(title="Tiempo"),
        yaxis2=attr(title="Fracción de Consumo"),
        xaxis3=attr(title="Tiempo"),
        yaxis3=attr(title="Consumo Acumulado"),
        title="Gráficos Interactivos del Problema de la Torta"
    )

    # Mostrar los gráficos en una sola figura interactiva
    plot([plot_consumo_flujo, plot_fraccion_consumo, plot_consumo_acumulado], layout)
    
    # Agregar el gráfico adicional area2() al final
    area2()
end

# Llamar a la función para simular el problema de la torta y mostrar los gráficos con PlotlyJS
simular_torta_plotlyjs(ce_default, 100, 1)



using PlotlyJS

function area2_stacked()
    function _stacked_area!(traces)
        for (i, tr) in enumerate(traces[2:end])
            for j in 1:min(length(traces[i]["y"]), length(tr["y"]))
                tr["y"][j] += traces[i]["y"][j]
            end
        end
        traces
    end

    traces = [scatter(;x=1:3, y=[2, 1, 4], fill="tozeroy"),
              scatter(;x=1:3, y=[1, 1, 2], fill="tonexty"),
              scatter(;x=1:3, y=[3, 0, 2], fill="tonexty")]
    _stacked_area!(traces)

    plot(traces, Layout(title="Gráfico de Área Apilada"))
end

# Llamar a la función para generar el gráfico de área apilada
area2_stacked()

include("cakeeating.jl")
include("itpcake.jl")

using PlotlyJS

function simular_torta_plotlyjs(ce_default, T, k0)
    # Inicializar vectores para guardar las sucesiones de consumo y capital
    C = zeros(T+1)
    K = zeros(T+1)

    # Inicializar el valor de la torta
    torta = zeros(T+1)

    # Establecer el estado inicial
    C[1] = k0
    K[1] = k0
    torta[1] = k0  # La torta inicial es el capital inicial

    # Crear interpoladores para las funciones de consumo y ahorro
    interpolate_consumo1(k) = interpolate([ce_default.kgrid, ce_default.gc],Gridded(Linear()))(k)
    interpolate_ahorro(k) = interpolate([ce_.kgrid, ce_default.gk],Gridded(Linear()))(k)
    # Simular el comportamiento a lo largo del tiempo
    for t in 1:T
        k = K[t]

        # Usar el interpolador para obtener el consumo en el estado actual
        C[t] = interpolate_consumo1(k)

        # Usar el interpolador para obtener el capital en el próximo período
        K[t+1] = interpolate_ahorro(k)

        # Calcular la torta restante
        torta[t] = K[t+1] - C[t]
    end

    # Crear un gráfico interactivo de área apilada para representar la torta y el consumo
    trace_torta = scatter(
        x=0:T, y=torta, fill="tozeroy", name="Torta", xaxis="x1", yaxis="y1"
    )

    trace_consumo = scatter(
        x=0:T, y=C, fill="tozeroy", name="Consumo", xaxis="x1", yaxis="y1"
    )

    # Crear un layout
    layout = Layout(
        title="Problema de la Torta",
        xaxis1=attr(title="Tiempo"),
        yaxis1=attr(title="Cantidad"),
        legend=attr(x=0.8, y=1),
    )

    # Mostrar el gráfico de área apilada
    plot([trace_torta, trace_consumo], layout)
end

# Llamar a la función para simular el problema de la torta y mostrar los gráficos con PlotlyJS
simular_torta_plotlyjs(ce_default, 100, 1)
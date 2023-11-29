using Pkg
Pkg.add("QuantEcon")
using QuantEcon, Optim, Interpolations, LinearAlgebra, PlotlyJS, Distributions

abstract type Deuda end
#Como reutiliza el codigo para los dos se arma el tipo deuda y dos subtipos de estructura una para el NoDefault y otra para el Arellano
struct NoDefault <: Deuda
    pars::Dict{Symbol,Float64}

    bgrid::Vector{Float64} #Bonos
    ygrid::Vector{Float64} #Ingresos es una cadena de markov
    Py::Matrix{Float64} #Probabilidad de pasar de uno a otro

    v::Matrix{Float64}

    gc::Matrix{Float64} #Política de consumo
    gb::Matrix{Float64} #Política de emisión de bonos
end

function NoDefault(;
    β=0.96,
    γ=2,
    r=0.017, #Tasa libre de riesgo
    ρy=0.945, #Persistencia del AR1 para el ingreso
    σy=0.025, #Volatibilidad del AR1 para el ingreso
    Nb=200, #puntos para los bonos
    Ny=21, #puntos para el ingreso
    bmax=0.9 #nivel maximo de los bonos
)

    pars = Dict(:β => β, :γ => γ, :r => r, :ρy => ρy, :σy => σy, :Nb => Nb, :Ny => Ny, :bmax => bmax)
#Convierte todo a flotante despues vemos que hacemos con esto
    ychain = tauchen(Ny, ρy, σy, 0, 2) #Esto es para generar la cadena de markov mediante un AR1, por eso cargamos el paquete QuantEcon
#μ = 0 y sd = 2

    Py = ychain.p #Matriz de transición
    ygrid = exp.(ychain.state_values) #Puntos, son el log de y por eso tomo la exponencial

    bgrid = range(0, bmax, length=Nb)

    v = zeros(Nb, Ny)
    gc = zeros(Nb, Ny)
    gb = zeros(Nb, Ny)

    return NoDefault(pars, bgrid, ygrid, Py, v, gc, gb)
end
#En el problema de la torta habiamos resuelto todo con no te dejo tener consumos negativos pero acá le pongo un mínimo pero sino cumple el mínimo
#Lo voy a calcular con la derivada de la función de utilidad en el mínimo (Aproximación de Taylor)
u(cv, dd::Deuda) = u(cv, dd.pars[:γ])
function u(cv, γ::Real)
    cmin = 1e-3
    if cv < cmin
        # Por debajo de cmin, lineal con la derivada de u en cmin. Aproximación de Taylor
        return u(cmin, γ) + (cv - cmin) * cmin^(-γ)
    else
        if γ == 1
            return log(cv)
        else
            return cv^(1 - γ) / (1 - γ)
        end
    end
end

# consumo es ingreso más ingresos por vender deuda nueva menos repago de deuda vieja
budget_constraint(bpv, bv, yv, q, dd::Deuda) = yv + q * bpv - bv

# LA MAGIA DEL MULTIPLE DISPATCH
debtprice(dd::NoDefault, bpv, yv, itp_q) = 1/(1+dd.pars[:r])

function eval_value(jb, jy, bpv, itp_q, itp_v, dd::Deuda) #Estamos en un b y elegimos un b'
    """ Evalúa la función de valor en (b,y) para una elección de b' """
    β = dd.pars[:β]
    bv, yv = dd.bgrid[jb], dd.ygrid[jy] #Punto donde estamos

    # Interpola el precio de la deuda para el nivel elegido
    qv = debtprice(dd, bpv, yv, itp_q) 

    # Deduce consumo del estado, la elección de deuda nueva y el precio de la deuda nueva
    cv = budget_constraint(bpv, bv, yv, qv, dd)

    # Evalúa la función de utilidad en c
    ut = u(cv, dd) 

    # Calcula el valor esperado de la función de valor interpolando en b' 
    Ev = 0.0
    #Recorremos todos los estados posibles viendo cual es la probabilidad de cada estado y sumando la probabilidad por
    #el valor de la función de valor en ese estado. Cuando terminemos de recorrer todo la suma del valor por la probabilidad
    #nos da el valor esperado de la función de valor. 
    for (jyp, ypv) in enumerate(dd.ygrid) #Son los de mañana
        prob = dd.Py[jy, jyp] #Probabilidad de pasar de y a yp
        Ev += prob * itp_v(bpv, ypv) #interpolo porque la bp esta fuera de la grilla, lo estoy eligiendo de manera continua. 
    #Estoy interpolando las dos variables pero yp no haria falta porque esta dentro de la grilla. 
    end

    # v es el flujo de hoy más el valor de continuación esperado descontado
    v = ut + β * Ev

    return v, cv
end

function opt_value(jb, jy, itp_q, itp_v, dd::Deuda) #Esto es para ambos, yo puedo reutilizar esto para despues.
    """ Elige b' en (b,y) para maximizar la función de valor """

    # b' ∈ bgrid
    b_min, b_max = extrema(dd.bgrid) #Extremos de la grilla

    # Función objetivo en términos de b'
    obj_f(bpv) = eval_value(jb, jy, bpv, itp_q, itp_v, dd)[1]

    # Resuelve el máximo
    res = Optim.maximize(obj_f, b_min, b_max) 

    # Extrae el argmax
    b_star = Optim.maximizer(res) 

    # Extrae v y c consistentes con b'
    vp, c_star = eval_value(jb, jy, b_star, itp_q, itp_v, dd) 

    return vp, c_star, b_star
end

function vfi_iter!(new_v, itp_q, dd::NoDefault) #Esto es solo para sin default
    # Reconstruye la interpolación de la función de valor
    knts = (dd.bgrid, dd.ygrid)
    itp_v = interpolate(knts, dd.v, Gridded(Linear()))

    for jy in eachindex(dd.ygrid), jb in eachindex(dd.bgrid)

        vp, c_star, b_star = opt_value(jb, jy, itp_q, itp_v, dd)

        # Guarda los valores para repago 
        new_v[jb, jy] = vp
        dd.gb[jb, jy] = b_star
        dd.gc[jb, jy] = c_star
    end
end

# LA MAGIA DE MULTIPLE DISPATCH
make_itp(dd::NoDefault) = 1/(1+dd.pars[:r]) #Esto es solo para default

function vfi!(dd::Deuda; tol::Float64=1e-8, maxiter=5000, verbose=true) #Esta VFI es para los dos casos, con y sin deuda porque dd::Deuda
    """ Itera sobre la ecuación de Bellman del país para encontrar la función de valor, probabilidad de default, consumo en repago y en default """
    new_v = similar(dd.v)

    dist = 1 + tol
    iter = 0

    # Interpolación del precio de la deuda (si hace falta)
    itp_q = make_itp(dd)

    # Loop principal sobre la Bellman del país
    while dist > tol && iter < maxiter
        iter += 1

        vfi_iter!(new_v, itp_q, dd)

        # Distancia entre la función de valor y el guess viejo 
        dist = norm(new_v - dd.v) / (1 + norm(dd.v))

        # Actualiza la función de valor 
        dd.v .= new_v
        verbose && print("Iteration $iter. Distance = $dist\n")
    end
    dist < tol && print("✓")
end






p = plot(scatter(x=ddd.bgrid, y= (ddd.gc[:, 5]/dd.ygrid[5])))



#Hago un gráfico normal 
cvec=range(0.01,length=1000)

plot(scatter(x=cvec, y=u.(cvec,2)))



dd=NoDefault()
u(-3,dd)

#Linea 135
make_itp(dd)

#Veamos cuanto nos da el problema, uso el vfi 
vfi!(dd)

#Podemos hacer un gráfico para ver como esta persona elige las cosas. 
propension=dd.gc[:,1]./dd.ygrid[:,1]

#Esto es para ver como es la función de consumo con diferentes valores
plot(scatter(x=dd.ygrid, y=dd.gc[:,1]./dd.ygrid[:,1],xaxis_title="Ingreso",yaxis_title="Propensión del consumo (gc(b, y)/y)"))


#Me voy a hacer un vector de varios scatter para ver como es la función de consumo 
plot([scatter(x=dd.bgrid, y=dd.gc[:,jy]/dd.ygrid, name="y = $yv",xaxis_title="Bono (b)",yaxis_title="Ingreso (y)") for (jy, yv) in enumerate(dd.ygrid)]) #Esto es para ver como es la función de consumo con diferentes valores

# Podemos ver que para un nivel de ingreso, mientras más deuda tenes menos podes consumir. Para niveles bajos de ingreso la función es más concava
#Esto es porque vos tenes dos motivos para endeudarte, por el beta y suavizar consumo. Cuando estas muy cerca del limite
#Sabes que no podes endeudarte más mañana entonces si mañana toca un shock malo queres endedudarte más mañana
#Entonces te empezas a cubrir antes mientras menor ingreso tengas entonces la función de consumo es más concava.


plot([])
#Ejercicio 1. Funciones de consumo 

#Este es el gráfico básico 

Ejercicio31=plot([scatter(x=dd.bgrid, y=dd.gc[:,jy]./ygrid, name="y = $yv") for (jy, yv) in enumerate(dd.ygrid)])
#
# Supongamos que dd es una instancia de NoDefault ya resuelta
bb, yy = dd.bgrid, dd.ygrid
consumption_grid = zeros(length(bb), length(yy))
for (jy, y) in enumerate(yy)
    for (jb, b) in enumerate(bb)
        _, consumption_grid[jb, jy] = opt_value(jb, jy, make_itp(dd), interpolate((bb, yy), dd.v, Gridded(Linear())), dd)
    end
end

# # Crear un gráfico de contorno para c(b, y)
# Crear un gráfico de contorno para c(b, y)
contour_plot = contour(
    x=bb,
    y=yy,
    z=consumption_grid',
    showscale=true
)

layout = Layout(
    xaxis_title="Bono (b)",
    yaxis_title="Ingreso (y)",
    title="Función de Consumo c(b, y)"
)

Ejercicio11=plot(contour_plot, layout)

savefig(Ejercicio11,"Ejercicio11.png")

#Esta raro el gráfico este pero revisar despues

#Vamos con el otro gráfico
bb, yy = dd.bgrid, dd.ygrid
propensity_grid = zeros(length(bb), length(yy))

for (jy, y) in enumerate(yy)
    for (jb, b) in enumerate(bb)
        _, consumption = opt_value(jb, jy, make_itp(dd), interpolate((bb, yy), dd.v, Gridded(Linear())), dd)
        propensity_grid[jb, jy] = consumption / y
    end
end


# Crear un gráfico de contorno para c(b, y)/y
contour_plot_propension = contour(
    x=bb,
    y=yy,
    z=propensity_grid',
    showscale=true
)


# Añadir título y etiquetas de ejes
layout = Layout(
    xaxis_title="Bono (b)",
    yaxis_title="Ingreso (y)",
    title="Propensión al Consumo c(b, y) / y"
)


Ejercicio12= plot(contour_plot_propension, layout)

savefig(Ejercicio12,"Ejercicio12.png")
## Revisar los gráficos

plot([Ejercicio11,Ejercicio12],layout=Layout(grid=Dict(:rows=>1,:columns=>2)))

p1=[Ejercicio11;
Ejercicio12]
#Ejercicio 2
#Vamos primero con q 
Nq=3
Q=range(0.96,1.0,length=Nq)
# Vector para guardar los gráficos de dispersión
pv = Vector{AbstractTrace}(undef, Nq)

# Modelo NoDefault con deuda en 0
dd_no_debt = NoDefault()

# Calcular la política de consumo para diferentes valores de q
for (i, q) in enumerate(Q)
    # Calcular r correspondiente a q
    r = 1 / q - 1

    # Preparar el modelo NoDefault con q y r
    dd = NoDefault(β=dd_no_debt.pars[:β], γ=dd_no_debt.pars[:γ], r=r, ρy=dd_no_debt.pars[:ρy], σy=dd_no_debt.pars[:σy])

    # Resolver el modelo
    vfi!(dd)

    # Calcular la política de consumo gc(0, y) para este valor de q
    consumption = dd.gc[1, :]

    # Crear el gráfico de dispersión
    scatter_plot = scatter(x=dd.ygrid, y=consumption, mode="lines+markers", name="q = $q")

    # Agregar el gráfico al vector pv
    pv[i] = scatter_plot
end

# Crear el gráfico final con los gráficos de dispersión
layout = Layout(
    xaxis_title="Ingreso (y)",
    yaxis_title="Consumo (gc(0, y))",
    title="Consumo vs. Ingreso para diferentes valores de q"
)

Ejercicio21_consumption=plot(pv, layout)

savefig(Ejercicio21_consumption,"Ejercicio21.png")
#Ahora con propensity consumption
for (i, q) in enumerate(Q)
    # Calcular r correspondiente a q
    r = 1 / q - 1

    # Preparar el modelo NoDefault con q y r
    dd = NoDefault(β=dd_no_debt.pars[:β], γ=dd_no_debt.pars[:γ], r=r, ρy=dd_no_debt.pars[:ρy], σy=dd_no_debt.pars[:σy])

    # Resolver el modelo
    vfi!(dd)

    # Calcular la política de consumo gc(0, y) para este valor de q
    propensityconsumption = dd.gc[1,:]./dd.ygrid

    # Crear el gráfico de dispersión
    scatter_plot = scatter(x=dd.ygrid, y=propensityconsumption, mode="lines+markers", name="q = $q")

    # Agregar el gráfico al vector pv
    pv[i] = scatter_plot
end

# Crear el gráfico final con los gráficos de dispersión
layout2 = Layout(
    xaxis_title="Ingreso (y)",
    yaxis_title="Propensión del consumo (gc(0, y)/y)",
    title="Propensión del consumo vs. Ingreso para diferentes valores de q"
)
Ejercicio21_pconsumption=plot(pv, layout2)

savefig(Ejercicio21_pconsumption,"Ejercicio21propensityq.png")


d=[Ejercicio21_consumption; Ejercicio21_pconsumption]


#Ejercicio 2: Análisis de σy
Nσy = 3
σy_values = range(0.01, 0.1, length=Nσy)

# Vector para guardar los gráficos de dispersión
pv_sigma = Vector{AbstractTrace}(undef, Nσy)

# Modelo NoDefault con deuda en 0
dd_no_debt = NoDefault()



# Para cada valor de σy en σy_values
for (i, σy) in enumerate(σy_values)
    # Preparar el modelo NoDefault con el valor de σy
    dd = NoDefault(β=dd_no_debt.pars[:β], γ=dd_no_debt.pars[:γ], r=dd_no_debt.pars[:r], ρy=dd_no_debt.pars[:ρy], σy=σy)


    # Resolver el modelo
    vfi!(dd)
    
    # Calcular la política de consumo gc(0, y)
    consumption = dd.gc[1, :]  # Consumo para b = 0
  # Crear el gráfico de dispersión
    scatter_plot = scatter(x=dd.ygrid, y=consumption, mode="lines+markers", name="σy = $σy")
    
    # Agregar el gráfico al vector pv_sigma
    pv_sigma[i] = scatter_plot
end


# Crear el gráfico final con los gráficos de dispersión
layout_sigma = Layout(
    xaxis_title="Ingreso (y)",
    yaxis_title="Consumo (gc(0, y))",
    title="Consumo vs. Ingreso para diferentes valores de σy"
)

Ejercicio21_sigmacon=plot(pv_sigma, layout_sigma)
savefig(Ejercicio21_sigmacon,"Ejercicio21_sigmacon.png")

Nσy = 3
σy_values = range(0.01, 0.1, length=Nσy)

# Vector para guardar los gráficos de dispersión
pv_sigma = Vector{AbstractTrace}(undef, Nσy)

# Modelo NoDefault con deuda en 0
dd_no_debt = NoDefault()


#Ahora con propensión al consumo. 
for (i, σy) in enumerate(σy_values)
    # Preparar el modelo NoDefault con el valor de σy
    dd = NoDefault(β=dd_no_debt.pars[:β], γ=dd_no_debt.pars[:γ], r=dd_no_debt.pars[:r], ρy=dd_no_debt.pars[:ρy], σy=σy)


    # Resolver el modelo
    vfi!(dd)
    
    # Calcular la política de consumo gc(0, y)
    propensityconsumption = dd.gc[1,:]./dd.ygrid

    # Crear el gráfico de dispersión
    scatter_plot = scatter(x=dd.ygrid, y=propensityconsumption, mode="lines+markers", name="σy = $σy")

    # Agregar el gráfico al vector pv
    pv[i] = scatter_plot
end


# Crear el gráfico final con los gráficos de dispersión
layout_sigma2 = Layout(
    xaxis_title="Ingreso (y)",
    yaxis_title="Propensión del consumo (gc(0, y)/y)",
    title="Propensión del Consumo vs. Ingreso para diferentes valores de σy"
)

plot(pv, layout_sigma2)

Ejercicio21_sigmacprop=plot(pv, layout_sigma)
savefig(Ejercicio21_sigmacprop,"Ejercicio21_sigmapropc.png")






#Ejercicio Opcional- Robutness 

abstract type Deuda end
#Como reutiliza el codigo para los dos se arma el tipo deuda y dos subtipos de estructura una para el NoDefault y otra para el Arellano
struct NoDefault <: Deuda
    pars::Dict{Symbol,Float64}

    bgrid::Vector{Float64} #Bonos
    ygrid::Vector{Float64} #Ingresos es una cadena de markov
    Py::Matrix{Float64} #Probabilidad de pasar de uno a otro

    v::Matrix{Float64}

    gc::Matrix{Float64} #Política de consumo
    gb::Matrix{Float64} #Política de emisión de bonos
end

function NoDefault(;
    β=0.96,
    γ=2,
    r=0.017, #Tasa libre de riesgo
    ρy=0.945, #Persistencia del AR1 para el ingreso
    σy=0.025, #Volatibilidad del AR1 para el ingreso
    Nb=200, #puntos para los bonos
    Ny=21, #puntos para el ingreso
    bmax=0.9, #nivel maximo de los bonos
    θ=0.1
    )
    pars = Dict(:β => β, :γ => γ, :r => r, :ρy => ρy, :σy => σy, :Nb => Nb, :Ny => Ny, :bmax => bmax, :θ => θ)
#Convierte todo a flotante despues vemos que hacemos con esto
    ychain = tauchen(Ny, ρy, σy, 0, 2) #Esto es para generar la cadena de markov mediante un AR1, por eso cargamos el paquete QuantEcon
#μ = 0 y sd = 2

    Py = ychain.p #Matriz de transición
    ygrid = exp.(ychain.state_values) #Puntos, son el log de y por eso tomo la exponencial

    bgrid = range(0, bmax, length=Nb)

    v = zeros(Nb, Ny)
    gc = zeros(Nb, Ny)
    gb = zeros(Nb, Ny)

    return NoDefault(pars, bgrid, ygrid, Py, v, gc, gb)
end
#En el problema de la torta habiamos resuelto todo con no te dejo tener consumos negativos pero acá le pongo un mínimo pero sino cumple el mínimo
#Lo voy a calcular con la derivada de la función de utilidad en el mínimo (Aproximación de Taylor)
u(cv, dd::Deuda) = u(cv, dd.pars[:γ])
function u(cv, γ::Real)
    cmin = 1e-3
    if cv < cmin
        # Por debajo de cmin, lineal con la derivada de u en cmin. Aproximación de Taylor
        return u(cmin, γ) + (cv - cmin) * cmin^(-γ)
    else
        if γ == 1
            return log(cv)
        else
            return cv^(1 - γ) / (1 - γ)
        end
    end
end

budget_constraint(bpv, bv, yv, q, dd::Deuda) = yv + q * bpv - bv

# LA MAGIA DEL MULTIPLE DISPATCH
debtprice(dd::NoDefault, bpv, yv, itp_q) = 1/(1+dd.pars[:r])


function eval_value(jb, jy, bpv, itp_q, itp_v, dd::Deuda, robust=true)
    """ Evalúa la función de valor en (b,y) para una elección de b' """
    β = dd.pars[:β]
    bv, yv = dd.bgrid[jb], dd.ygrid[jy]

    # Interpola el precio de la deuda para el nivel elegido
    qv = debtprice(dd, bpv, yv, itp_q)

    # Deduce consumo del estado, la elección de deuda nueva y el precio de la deuda nueva
    cv = budget_constraint(bpv, bv, yv, qv, dd)

    # Evalúa la función de utilidad en c
    ut = u(cv, dd)

    # Calcula el valor esperado de la función de valor interpolando en b'
    Ev = 0.0
    Ev2 = 0.0
    if robust
        for (jyp, ypv) in enumerate(dd.ygrid)
            prob = dd.Py[jy, jyp] 
            Ev2+=exp((-dd.pars[:θ])*itp_v(bpv, ypv))*prob
        end
		Ev = (-(1/dd.pars[:θ])) * log(Ev2)
    else 
        for (jyp, ypv) in enumerate(dd.ygrid)
            prob = dd.Py[jy, jyp]
            Ev += prob * itp_v(bpv, ypv)
        end
    end
    # v es el flujo de hoy más el valor de continuación esperado descontado
    v = ut + β * Ev

    return v, cv
end


function opt_value(jb, jy, itp_q, itp_v, dd::Deuda) #Esto es para ambos, yo puedo reutilizar esto para despues.
    """ Elige b' en (b,y) para maximizar la función de valor """

    # b' ∈ bgrid
    b_min, b_max = extrema(dd.bgrid) #Extremos de la grilla

    # Función objetivo en términos de b'
    obj_f(bpv) = eval_value(jb, jy, bpv, itp_q, itp_v, dd)[1]

    # Resuelve el máximo
    res = Optim.maximize(obj_f, b_min, b_max) 

    # Extrae el argmax
    b_star = Optim.maximizer(res) 

    # Extrae v y c consistentes con b'
    vp, c_star = eval_value(jb, jy, b_star, itp_q, itp_v, dd) 

    return vp, c_star, b_star
end

function vfi_iter!(new_v, itp_q, dd::NoDefault) #Esto es solo para sin default
    # Reconstruye la interpolación de la función de valor
    knts = (dd.bgrid, dd.ygrid)
    itp_v = interpolate(knts, dd.v, Gridded(Linear()))

    for jy in eachindex(dd.ygrid), jb in eachindex(dd.bgrid)

        vp, c_star, b_star = opt_value(jb, jy, itp_q, itp_v, dd)

        # Guarda los valores para repago 
        new_v[jb, jy] = vp
        dd.gb[jb, jy] = b_star
        dd.gc[jb, jy] = c_star
    end
end

# LA MAGIA DE MULTIPLE DISPATCH
make_itp(dd::NoDefault) = 1/(1+dd.pars[:r]) #Esto es solo para default

function vfi!(dd::Deuda; tol::Float64=1e-8, maxiter=5000, verbose=true) #Esta VFI es para los dos casos, con y sin deuda porque dd::Deuda
    """ Itera sobre la ecuación de Bellman del país para encontrar la función de valor, probabilidad de default, consumo en repago y en default """
    new_v = similar(dd.v)

    dist = 1 + tol
    iter = 0

    # Interpolación del precio de la deuda (si hace falta)
    itp_q = make_itp(dd)

    # Loop principal sobre la Bellman del país
    while dist > tol && iter < maxiter
        iter += 1

        vfi_iter!(new_v, itp_q, dd)

        # Distancia entre la función de valor y el guess viejo 
        dist = norm(new_v - dd.v) / (1 + norm(dd.v))

        # Actualiza la función de valor 
        dd.v .= new_v
        verbose && print("Iteration $iter. Distance = $dist\n")
    end
    dist < tol && print("✓")
end


#Gráfico

function punto3a(dd::NoDefault)
    theta = range(-1, 1, 3)
    pv3 = Vector{AbstractTrace}(undef, 8)
    for (qj, qv) in enumerate(theta)
        ddd = NoDefault(θ=qv)
        vfi!(ddd)
        #for (jy, yv) in enumerate(dd.ygrid)
        pv3[qj] = scatter(x=ddd.bgrid, y= (ddd.gc[:, 5]/dd.ygrid[5]), name = "θ = $qv")
        #end 
    end
    plot(pv3)
    layout = Layout(
        title = "Propención a consumir vs Deuda para distintos θ (Y=0.98)",
        xaxis_title = "Deuda",
        yaxis_title = "Propención a consumir"
       )
       p3 = plot(pv3, layout)
       savefig(p3, "Grafico3afinal.png")
    end

   
#Me armo otro pero para el modelo no robusto


abstract type Deuda end
#Como reutiliza el codigo para los dos se arma el tipo deuda y dos subtipos de estructura una para el NoDefault y otra para el Arellano
struct NoDefault <: Deuda
    pars::Dict{Symbol,Float64}

    bgrid::Vector{Float64} #Bonos
    ygrid::Vector{Float64} #Ingresos es una cadena de markov
    Py::Matrix{Float64} #Probabilidad de pasar de uno a otro

    v::Matrix{Float64}

    gc::Matrix{Float64} #Política de consumo
    gb::Matrix{Float64} #Política de emisión de bonos
end

function NoDefault(;
    β=0.96,
    γ=2,
    r=0.017, #Tasa libre de riesgo
    ρy=0.945, #Persistencia del AR1 para el ingreso
    σy=0.025, #Volatibilidad del AR1 para el ingreso
    Nb=200, #puntos para los bonos
    Ny=21, #puntos para el ingreso
    bmax=0.9, #nivel maximo de los bonos
    )
    pars = Dict(:β => β, :γ => γ, :r => r, :ρy => ρy, :σy => σy, :Nb => Nb, :Ny => Ny, :bmax => bmax)
#Convierte todo a flotante despues vemos que hacemos con esto
    ychain = tauchen(Ny, ρy, σy, 0, 2) #Esto es para generar la cadena de markov mediante un AR1, por eso cargamos el paquete QuantEcon
#μ = 0 y sd = 2

    Py = ychain.p #Matriz de transición
    ygrid = exp.(ychain.state_values) #Puntos, son el log de y por eso tomo la exponencial

    bgrid = range(0, bmax, length=Nb)

    v = zeros(Nb, Ny)
    gc = zeros(Nb, Ny)
    gb = zeros(Nb, Ny)

    return NoDefault(pars, bgrid, ygrid, Py, v, gc, gb)
end
#En el problema de la torta habiamos resuelto todo con no te dejo tener consumos negativos pero acá le pongo un mínimo pero sino cumple el mínimo
#Lo voy a calcular con la derivada de la función de utilidad en el mínimo (Aproximación de Taylor)
u(cv, dd::Deuda) = u(cv, dd.pars[:γ])
function u(cv, γ::Real)
    cmin = 1e-3
    if cv < cmin
        # Por debajo de cmin, lineal con la derivada de u en cmin. Aproximación de Taylor
        return u(cmin, γ) + (cv - cmin) * cmin^(-γ)
    else
        if γ == 1
            return log(cv)
        else
            return cv^(1 - γ) / (1 - γ)
        end
    end
end

budget_constraint(bpv, bv, yv, q, dd::Deuda) = yv + q * bpv - bv

# LA MAGIA DEL MULTIPLE DISPATCH
debtprice(dd::NoDefault, bpv, yv, itp_q) = 1/(1+dd.pars[:r])


function eval_value(jb, jy, bpv, itp_q, itp_v, dd::Deuda, robust=false)
    """ Evalúa la función de valor en (b,y) para una elección de b' """
    β = dd.pars[:β]
    bv, yv = dd.bgrid[jb], dd.ygrid[jy]

    # Interpola el precio de la deuda para el nivel elegido
    qv = debtprice(dd, bpv, yv, itp_q)

    # Deduce consumo del estado, la elección de deuda nueva y el precio de la deuda nueva
    cv = budget_constraint(bpv, bv, yv, qv, dd)

    # Evalúa la función de utilidad en c
    ut = u(cv, dd)

    # Calcula el valor esperado de la función de valor interpolando en b'
    Ev = 0.0
    Ev2 = 0.0
    if robust
        for (jyp, ypv) in enumerate(dd.ygrid)
            prob = dd.Py[jy, jyp] 
            Ev2+=exp((-dd.pars[:θ])*itp_v(bpv, ypv))*prob
        end
		Ev = (-(1/dd.pars[:θ])) * log(Ev2)
    else 
        for (jyp, ypv) in enumerate(dd.ygrid)
            prob = dd.Py[jy, jyp]
            Ev += prob * itp_v(bpv, ypv)
        end
    end
    # v es el flujo de hoy más el valor de continuación esperado descontado
    v = ut + β * Ev

    return v, cv
end


function opt_value(jb, jy, itp_q, itp_v, dd::Deuda) #Esto es para ambos, yo puedo reutilizar esto para despues.
    """ Elige b' en (b,y) para maximizar la función de valor """

    # b' ∈ bgrid
    b_min, b_max = extrema(dd.bgrid) #Extremos de la grilla

    # Función objetivo en términos de b'
    obj_f(bpv) = eval_value(jb, jy, bpv, itp_q, itp_v, dd)[1]

    # Resuelve el máximo
    res = Optim.maximize(obj_f, b_min, b_max) 

    # Extrae el argmax
    b_star = Optim.maximizer(res) 

    # Extrae v y c consistentes con b'
    vp, c_star = eval_value(jb, jy, b_star, itp_q, itp_v, dd) 

    return vp, c_star, b_star
end

function vfi_iter!(new_v, itp_q, dd::NoDefault) #Esto es solo para sin default
    # Reconstruye la interpolación de la función de valor
    knts = (dd.bgrid, dd.ygrid)
    itp_v = interpolate(knts, dd.v, Gridded(Linear()))

    for jy in eachindex(dd.ygrid), jb in eachindex(dd.bgrid)

        vp, c_star, b_star = opt_value(jb, jy, itp_q, itp_v, dd)

        # Guarda los valores para repago 
        new_v[jb, jy] = vp
        dd.gb[jb, jy] = b_star
        dd.gc[jb, jy] = c_star
    end
end

# LA MAGIA DE MULTIPLE DISPATCH
make_itp(dd::NoDefault) = 1/(1+dd.pars[:r]) #Esto es solo para default

function vfi!(dd::Deuda; tol::Float64=1e-8, maxiter=5000, verbose=true) #Esta VFI es para los dos casos, con y sin deuda porque dd::Deuda
    """ Itera sobre la ecuación de Bellman del país para encontrar la función de valor, probabilidad de default, consumo en repago y en default """
    new_v = similar(dd.v)

    dist = 1 + tol
    iter = 0

    # Interpolación del precio de la deuda (si hace falta)
    itp_q = make_itp(dd)

    # Loop principal sobre la Bellman del país
    while dist > tol && iter < maxiter
        iter += 1

        vfi_iter!(new_v, itp_q, dd)

        # Distancia entre la función de valor y el guess viejo 
        dist = norm(new_v - dd.v) / (1 + norm(dd.v))

        # Actualiza la función de valor 
        dd.v .= new_v
        verbose && print("Iteration $iter. Distance = $dist\n")
    end
    dist < tol && print("✓")
end












#Gráfico 







# Define el valor de y que deseas utilizar para la comparación
y_value = 1.0  # Cambia esto al valor de y que te interese

# Resuelve el modelo original
dd_original = NoDefault2()
vfi!(dd_original)

# Resuelve el modelo con robustez
dd_robust = NoDefault(θ=5)  
vfi!(dd_robust)

# Calcula la propensión marginal al consumo en equilibrio parcial para el modelo original
propensity_original = dd_original.gc[:, findfirst(dd_original.ygrid .≈ y_value)]

# Calcula la propensión marginal al consumo en equilibrio parcial para el modelo con robustez
propensity_robust = dd_robust.gc[:, findfirst(dd_robust.ygrid .≈ y_value)]

# Crea un gráfico de dispersión para comparar las propensiones marginales al consumo
trace_original = scatter(x=dd_original.bgrid, y=propensity_original ./ y_value, mode="lines+markers", name="Original")
trace_robust = scatter(x=dd_robust.bgrid, y=propensity_robust ./ y_value, mode="lines+markers", name="Robusto")

data = [trace_original, trace_robust]

layout = Layout(
    xaxis_title="Bono (b)",
    yaxis_title="Propensión Marginal al Consumo (c(b, y)/y)",
    title="Comparación de Propensión Marginal al Consumo (Equilibrio Parcial)"
)

plot(data, layout)



























#simulador
using PlotlyJS, Random

function iter_simul_deuda(bt, yt, itp_gc, itp_gb, ymin, ymax, ρ, σ)
    ct = itp_gc(bt, yt)

    bp = itp_gb(bt, yt)

    ϵt = rand(Normal(0,1))
	yp = exp(ρ * log(yt) + σ * ϵt)

    yp = max(min(ymax, yp), ymin)

    return bp, yp, ct
end

function simul(dd::NoDefault; b0 = mean(dd.bgrid), y0 = mean(dd.ygrid), T = 10000)    
    ymin, ymax = extrema(dd.ygrid)
    ρ = 0.8
    σ = 0.02
    bnots = (dd.bgrid, dd.ygrid)
    
    itp_gb = interpolate(bnots, dd.gb, Gridded(Linear()))
    itp_gc = interpolate(bnots, dd.gc, Gridded(Linear()))

    sample = Dict(sym => Vector{Float64}(undef, T) for sym in (:b, :y, :c))

    sample[:y][1] = y0
    sample[:b][1] = b0
    
    for jt in 1:T

        b0, y0, c = iter_simul_deuda(b0, y0, itp_gc, itp_gb, ymin, ymax, ρ, σ)

        sample[:c][jt] = c

        if jt < T
            sample[:b][jt+1] = b0
            sample[:y][jt+1] = y0
        end
    end

    return sample
end



dd=NoDefault(;
β=0.96,
γ=2,
r=0.017, #Tasa libre de riesgo
ρy=0.945, #Persistencia del AR1 para el ingreso
σy=0.025, #Volatibilidad del AR1 para el ingreso
Nb=200, #puntos para los bonos
Ny=21, #puntos para el ingreso
bmax=0.9, #nivel maximo de los bonos
θ=0.5
)

vfi!(dd; tol = 1e-8, maxiter = 2000, verbose = true)

simulacion = simul(dd, b0 = mean(dd.bgrid), y0 = mean(dd.ygrid), T = 10000)


cy_sim = simulacion[:c] ./ simulacion[:y]
b_sim = simulacion[:b]

plot([scatter(x=1:length(cy_sim), y=cy_sim, mode="lines+markers", name="c/y"), scatter(x=1:length(b_sim), y=b_sim, mode="lines+markers", name="b")])

#Veo las distribuciones de deuda y consumo
a=histogram_cy_sim = plot(PlotlyJS.histogram(x = cy_sim, nbinsx = 10, name = "Distribución de c/y simulado"),
Layout(
	title = "Distribución de c/y simulado",
	xaxis_range = [1, 1.5],
	yaxis_range = [0, 5],
	xaxis_title = "Valor de c/y",
	yaxis_title = "Frecuencia"
))
savefig(a,"histogram_cy_robust0.5.png")

b=histogram_b_sim = plot(PlotlyJS.histogram(x = simulacion[:b], nbinsx = 10, name = "Distribución de b simulado"),
Layout(
	title = "Distribución de b simulado",
	xaxis_range = [0, 1],
	yaxis_range = [0, 5],
	xaxis_title = "Valor de b",
	yaxis_title = "Frecuencia"
)
)

savefig(b,"histogram_b_robust0.5.png")



#Ahora vamos con un theta de 5 y 1

dd2=NoDefault(;
β=0.96,
γ=2,
r=0.017, #Tasa libre de riesgo
ρy=0.945, #Persistencia del AR1 para el ingreso
σy=0.025, #Volatibilidad del AR1 para el ingreso
Nb=200, #puntos para los bonos
Ny=21, #puntos para el ingreso
bmax=0.9, #nivel maximo de los bonos
θ=5
)

vfi!(dd2; tol = 1e-8, maxiter = 2000, verbose = true)

simulacion = simul(dd2, b0 = mean(dd.bgrid), y0 = mean(dd.ygrid), T = 10000)


cy_sim = simulacion[:c] ./ simulacion[:y]
b_sim = simulacion[:b]

plot([scatter(x=1:length(cy_sim), y=cy_sim, mode="lines+markers", name="c/y"), scatter(x=1:length(b_sim), y=b_sim, mode="lines+markers", name="b")])

#Veo las distribuciones de deuda y consumo
a=histogram_cy_sim = plot(PlotlyJS.histogram(x = cy_sim, nbinsx = 10, name = "Distribución de c/y simulado"),
Layout(
	title = "Distribución de c/y simulado",
	xaxis_range = [1, 1.5],
	yaxis_range = [0, 5],
	xaxis_title = "Valor de c/y",
	yaxis_title = "Frecuencia"
))
savefig(a,"histogram_cy_robust1.png")

b=histogram_b_sim = plot(PlotlyJS.histogram(x = simulacion[:b], nbinsx = 10, name = "Distribución de b simulado"),
Layout(
	title = "Distribución de b simulado",
	xaxis_range = [0, 1],
	yaxis_range = [0, 5],
	xaxis_title = "Valor de b",
	yaxis_title = "Frecuencia"
)
)

savefig(b,"histogram_b_robust1.png")

p=[histogram_cy_sim;histogram_b_sim]

savefig(p,"histogram_robust5conjunto.png")
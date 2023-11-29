include("arellano.jl")


abstract type Costo end
abstract type OG <: Costo end
abstract type Lin <: Costo end
abstract type Quad <: Costo end


struct DeudaLarga{T<:Costo} <: Default
    pars::Dict{Symbol,Float64}

    bgrid::Vector{Float64}
    ygrid::Vector{Float64}
    Py::Matrix{Float64}

    v::Matrix{Float64}
    vR::Matrix{Float64}
    vD::Matrix{Float64}
    prob::Matrix{Float64}

    gc::Array{Float64,3}
    gb::Array{Float64,2}

    q::Array{Float64, 3}
end

function switch_ρ!(dd::DeudaLarga, ρ)
    dd.pars[:κ] = dd.pars[:r]+ρ
    dd.pars[:ρ] = ρ
    nothing
end

function DeudaLarga(; T = OG,
    β=0.953,
    γ=2,
    r=0.017,
    ψ=0.282,
    χ=0.01,
    Δ=0.1,
    d0=0.1,
    d1=0,
    ℏ=0.4,
    ρy=0.945,
    σy=0.025,
    Nb=200,
    Ny=21,
    ρ = 0.05,
    bmax=1.5,
)
    κ = r+ρ

    pars = Dict(:β => β, :γ => γ, :r => r, :ψ => ψ, :χ => χ, :ρy => ρy, :σy => σy, :κ => κ, :ρ => ρ, :ℏ => ℏ)

    if T == Lin
        pars[:Δ] = Δ
    elseif T == Quad
        pars[:d0] = d0
        pars[:d1] = d1
    end

    ychain = tauchen(Ny, ρy, σy, 0, 2)

    Py = ychain.p
    ygrid = exp.(ychain.state_values)

    bgrid = range(0, bmax, length=Nb)

    v = zeros(Nb, Ny)
    vR = zeros(Nb, Ny)
    vD = zeros(Nb, Ny)
    prob = zeros(Nb, Ny)

    gc = zeros(Nb, Ny, 2)
    gb = zeros(Nb, Ny)

    q  = ones(Nb, Ny, 2)

    return DeudaLarga{T}(pars, bgrid, ygrid, Py, v, vR, vD, prob, gc, gb, q)
end

function switch_Costo(dd::DeudaLarga, T; Δ = 0.1, d0 = 0.1, d1 = 0)

    d2 = DeudaLarga{T}(copy(dd.pars), copy(dd.bgrid), copy(dd.ygrid), copy(dd.Py), copy(dd.v), copy(dd.vR), copy(dd.vD), copy(dd.prob), copy(dd.gc), copy(dd.gb), copy(dd.q))

    if T == Lin
        d2.pars[:Δ] = Δ
    elseif T == Quad
        d2.pars[:d0] = d0
        d2.pars[:d1] = d1
    end

	return d2
end

h(yv, dd::DeudaLarga{OG}) = defcost_OG(yv)
h(yv, dd::DeudaLarga{Lin}) = defcost_lineal(yv, dd)
h(yv, dd::DeudaLarga{Quad}) = defcost_quad(yv, dd)


function budget_constraint(bpv, bv, yv, q, dd::DeudaLarga)
    κ, ρ = dd.pars[:κ], dd.pars[:ρ]
    # consumo es ingreso más ingresos por vender deuda nueva menos repago de deuda vieja
    cv = yv + q * (bpv - (1-ρ) * bv) - κ * bv
    return cv
end

function value_default(jb, jy, dd::DeudaLarga)
    """ Calcula el valor de estar en default en el estado (b,y) """
    β, ψ = (dd.pars[sym] for sym in (:β, :ψ))
    yv = dd.ygrid[jy]

    # Consumo en default es el ingreso menos los costos de default
    c = h(yv, dd)

    # Valor de continuación tiene en cuenta la probabilidad ψ de reacceder a mercados
    Ev = 0.0
    for jyp in eachindex(dd.ygrid)
        prob = dd.Py[jy, jyp]
        Ev += prob * (ψ * dd.v[jb, jyp] + (1 - ψ) * dd.vD[jb, jyp])
    end

    v = u(c, dd) + β * Ev

    return c, v
end

function vfi_iter!(new_v, itp_q, dd::DeudaLarga)
    # Reconstruye la interpolación de la función de valor
    itp_v = make_itp(dd, dd.v)

    for jy in eachindex(dd.ygrid)
        for jb in eachindex(dd.bgrid)

            # En repago
            vp, c_star, b_star = opt_value(jb, jy, itp_q, itp_v, dd)

            # Guarda los valores para repago 
            dd.vR[jb, jy] = vp
            dd.gb[jb, jy] = b_star
            dd.gc[jb, jy, 1] = c_star

            # En default
            cD, vD = value_default(jb, jy, dd)
            dd.vD[jb, jy] = vD
            dd.gc[jb, jy, 2] = cD
        end
    end

    χ, ℏ = (dd.pars[key] for key in (:χ, :ℏ))
    itp_vD = make_itp(dd, dd.vD)
    for (jb,bv) in enumerate(dd.bgrid), (jy,yv) in enumerate(dd.ygrid)
        # Valor de repagar y defaultear llegando a (b,y)
        vr = dd.vR[jb, jy]
        vd = itp_vD(bv*(1-ℏ), yv)

        # Probabilidad de default
        ## Modo 2: valor extremo tipo X evitando comparar exponenciales de cosas grandes
        lse = logsumexp([vd / χ, vr / χ])
        lpr = vd / χ - lse
        pr = exp(lpr)
        V = χ * lse

        # Guarda el valor y la probabilidad de default al llegar a (b,y)
        new_v[jb, jy] = V
        dd.prob[jb, jy] = pr
    end
end

make_itp(dd::DeudaLarga, y::Array{Float64,3}, jdef = 1) = make_itp(dd, y[:,:,jdef])

function q_iter!(new_q, dd::DeudaLarga)
    """ Ecuación de Euler de los acreedores determinan el precio de la deuda dada la deuda, el ingreso, y el precio esperado de la deuda """
    r, κ, ρ, ℏ, ψ = (dd.pars[key] for key in (:r, :κ, :ρ, :ℏ, :ψ))

    # 1 es repago, 2 es default
    itp_q = make_itp(dd, dd.q)

    for (jbp, bpv) in enumerate(dd.bgrid), jy in eachindex(dd.ygrid)
        Eq = 0.0
        EqD = 0.0
        for (jyp, ypv) in enumerate(dd.ygrid)
            prob = dd.Py[jy, jyp]
            prob_def = dd.prob[jbp, jyp]

            bpp = dd.gb[jbp, jyp]

            qp = itp_q(bpp, ypv, 1)

            R = κ + (1-ρ) * qp

            # Si el país tiene acceso a mercados, emite y puede hacer default mañana
            rep_R = (1 - prob_def) * R + prob_def * (1 - ℏ) * itp_q((1-ℏ)*bpv, ypv, 2)
            
            Eq += prob * rep_R
            EqD += prob * (ψ * rep_R + (1-ψ) * dd.q[jbp, jyp, 2])
        end
        new_q[jbp, jy, 1] = Eq  / (1+r)
        new_q[jbp, jy, 2] = EqD / (1+r)
    end
end















#Ejercicio 1. Probabilidad de default 
dd = DeudaLarga()

# Supongamos que dd es una instancia de NoDefault ya resuelta
bb, yy = dd.bgrid, dd.ygrid

probability_grid = zeros(length(bb), length(yy))
for (jy, y) in enumerate(yy)
    for (jb, b) in enumerate(bb)
        jdef = 1  # Asumiendo que jdef es 1
        y_matrix = ones(length(bb), length(yy), jdef)  # Crear una matriz tridimensional con dimensiones apropiadas
        _, probability_grid[jb, jy] = opt_value(jb, jy, make_itp(dd, y_matrix), interpolate((bb, yy), dd.v, Gridded(Linear())), dd)
    end
end


# # Crear un gráfico de contorno para c(b, y)
# Crear un gráfico de contorno para c(b, y)
contour_plot = contour(
    x=bb,
    y=yy,
    z=probability_grid',
    showscale=true
)

layout = Layout(
    xaxis_title="Bono (b)",
    yaxis_title="Ingreso (y)",
    title="Probabilidad de default"
)

Ejercicio11=plot(contour_plot, layout)

savefig(Ejercicio11,"Ejercicio11.png")


#Ejercicio 1 - Opcional
# Calcula el contour de probabilidad de default con los valores originales de beta y delta
dd_original = DeudaLarga()
probability_grid_original = zeros(length(bb), length(yy))
for (jy, y) in enumerate(yy)
    for (jb, b) in enumerate(bb)
        jdef = 1  # Asumiendo que jdef es 1
        y_matrix = ones(length(bb), length(yy), jdef)  # Crear una matriz tridimensional con dimensiones apropiadas
        _, probability_grid_original[jb, jy] = opt_value(jb, jy, make_itp(dd_original, y_matrix), interpolate((bb, yy), dd_original.v, Gridded(Linear())), dd_original)
    end
end
# Define los valores alternativos para beta y delta
beta_alternativo = 0.9
delta_alternativo = 0.2

# Modifica los parámetros de DeudaLarga con los valores alternativos
dd_beta_alternativo = DeudaLarga(β=beta_alternativo)
dd_delta_alternativo = DeudaLarga(Δ=delta_alternativo)

# Calcula el contour de probabilidad de default con los valores alternativos de beta y delta
probability_grid_beta_alternativo = zeros(length(bb), length(yy))
for (jy, y) in enumerate(yy)
    for (jb, b) in enumerate(bb)
        jdef = 1  # Asumiendo que jdef es 1
        y_matrix = ones(length(bb), length(yy), jdef)  # Crear una matriz tridimensional con dimensiones apropiadas
        _, probability_grid_beta_alternativo[jb, jy] = opt_value(jb, jy, make_itp(dd_beta_alternativo, y_matrix), interpolate((bb, yy), dd_beta_alternativo.v, Gridded(Linear())), dd_beta_alternativo)
    end
end

# Crea dos gráficos para comparar
contour_plot_original = contour(
    x=bb,
    y=yy,
    z=probability_grid_original',
    showscale=true,
    name="Original"
)

contour_plot_beta_alternativo = contour(
    x=bb,
    y=yy,
    z=probability_grid_beta_alternativo',
    showscale=true,
    name="Beta Alternativo"
)

layout = Layout(
    xaxis_title="Bono (b)",
    yaxis_title="Ingreso (y)",
    title="Comparación de Probabilidad de Default"
)

Ejercicio11 = plot([contour_plot_original, contour_plot_beta_alternativo], layout)
plot(contour_plot_beta_alternativo, layout)
savefig(Ejercicio11, "Ejercicio11_comparacion.png")


#Ejercicio 2

# Valores de χ
χ_values = [0.0001, 0.01, 0.1]

# Inicializa un arreglo para almacenar las instancias del modelo
arellano_models = []
dd = Arellano()
# Resuelve el modelo para cada valor de χ
for χ in χ_values
    # Crea una instancia del modelo Arellano con el valor de χ
    model = Arellano(χ=χ)

    # Resuelve el modelo
    eqm!(dd)

    # Almacena el modelo resuelto en el arreglo
    push!(arellano_models, model)
end

# Grafica las funciones de valor V, vR y vD para cada valor de χ
using Plots

# Nivel de deuda b
b_values = arellano_models[1].bgrid

# Valores de y
y_value = 1.0  # Mantén y fijo cerca de 1

# Inicializa los arreglos para almacenar los resultados
V_values = []
vR_values = []
vD_values = []

for model in arellano_models
    # Evalúa las funciones de valor para cada valor de b
    V = [value_default(findnearest(y_value, model.ygrid), model).v for b in b_values]
    vR = model.vR[:, findnearest(y_value, model.ygrid)]
    vD = model.vD[findnearest(y_value, model.ygrid)]

    push!(V_values, V)
    push!(vR_values, vR)
    push!(vD_values, vD)
end

# Grafica los resultados
plot(b_values, V_values, label=["χ = 0.0001" "χ = 0.01" "χ = 0.1"], xlabel="Nivel de Deuda (b)",
    ylabel="Valor", title="Funciones de Valor para distintos χ", lw=2)
plot!(b_values, vR_values, lw=2, linestyle=:dash, label="vR")
plot!(b_values, vD_values, lw=2, linestyle=:dot, label="vD")

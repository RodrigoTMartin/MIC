include("simulador_default.jl")
include("arellano.jl")


#========Regiones de repago y default=================#
sd1 = Arellano()

mpe!(sd1)

layout1 = Layout(title="Regiones de repago y default",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Income", 
yaxis_title = "Debt",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)


layout1 = Layout(
    xaxis_title="Bono (b)",
    yaxis_title="Ingreso (y)",
    title="Regiones de repago y default",
    paper_bgcolor="#272929", plot_bgcolor="#272929",
    xaxis=attr(zeroline=false, gridcolor="#434343"),
    yaxis=attr(zeroline=false, gridcolor="#434343"),
    font_color="#F8F1E9", font_size=12,
    font_family="Lato",
    hovermode="x",
)

repanddef = plot([contour(x = sd1.ygrid, y = sd1.bgrid, z = sd1.prob, showscale=true,colorbar=attr(title="Probabilidad de default",titleside="right"),name= "Regiones de repago y default",colorscale = "Plasma")], layout1)

savefig(repanddef, "repayanddefault.png")

#========Regiones de repago y default con cambios en β (opcional)=========#
sd2 = Arellano(β = 0.99)

mpe!(sd2)



layout2 = Layout(
    xaxis_title="Bono (b)",
    yaxis_title="Ingreso (y)",
    title="Regiones de repago y default con más impaciencia (β = 0.99)",
    paper_bgcolor="#272929", plot_bgcolor="#272929",
    xaxis=attr(zeroline=false, gridcolor="#434343"),
    yaxis=attr(zeroline=false, gridcolor="#434343"),
    font_color="#F8F1E9", font_size=12,
    font_family="Lato",
    hovermode="x",
)
repanddefβ = plot(contour(x = sd2.ygrid, y = sd2.bgrid, z = sd2.prob, colorbar=attr(title="Probabilidad de default",titleside="right"), contours_coloring="Plasma",colorscale = "Plasma"), layout2)

savefig(repanddefβ, "repayanddefaultbeta099.png")

#========Regiones de repago y default con cambios en Δ (opcional)=========#

sd3 = Arellano(Δ = 0.05)

mpe!(sd3)

layout3 = Layout(
    xaxis_title="Bono (b)",
    yaxis_title="Ingreso (y)",
    title="Regiones de repago y default con cambios en el costo de default (Δ = 0.05)",
    paper_bgcolor="#272929", plot_bgcolor="#272929",
    xaxis=attr(zeroline=false, gridcolor="#434343"),
    yaxis=attr(zeroline=false, gridcolor="#434343"),
    font_color="#F8F1E9", font_size=12,
    font_family="Lato",
    hovermode="x",
)

repanddefΔ = plot(contour(x = sd3.ygrid, y = sd3.bgrid, z = sd3.prob, colorbar=attr(title="Probabilidad de default",titleside="right"), contours_coloring="heatmap", colorscale = "Plasma"), layout3)

savefig(repanddefΔ, "repayanddefaultDelta.png")
#=====Mega opcional====#
unos3 = ones(200,1)

sd2.v
sd2.vD
sd2.vR
sd2.bgrid

modsd2vD = sd2.vD[11] * unos3

scatters2 = Vector{AbstractTrace}(undef, 3)
scatters2[1] = scatter(x = sd2.bgrid, y = sd2.v[:,11], name = "V", marker= attr(color="00FFFF"))
scatters2[2] = scatter(x = sd2.bgrid, y = modsd2vD[:,1], name = "vD", marker = attr(color="#8A2BE2"))
scatters2[3] = scatter(x = sd2.bgrid, y = sd2.vR[:,11], name = "vR",marker = attr(color = "#F8F1E9"))

layout6 = Layout(title="Funciones de valor con más impaciencia (β = 0.99)",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Deuda", 
yaxis_title = "Valor",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot6 = plot(scatters2, layout6)

savefig(plot6, "MuyOpcionalBeta.png")

unos4 = ones(200,1)

sd3.v
sd3.vD
sd3.vR
sd3.bgrid

modsd3vD = sd3.vD[11] * unos4

scatters3 = Vector{AbstractTrace}(undef, 3)
scatters3[1] = scatter(x = sd3.bgrid, y = sd3.v[:,11], name = "V", marker= attr(color="00FFFF"))
scatters3[2] = scatter(x = sd3.bgrid, y = modsd3vD[:,1], name = "vD",marker = attr(color="#8A2BE2"))
scatters3[3] = scatter(x = sd3.bgrid, y = sd3.vR[:,11], name = "vR",marker = attr(color = "#F8F1E9"))

layout7 = Layout(title="Funciones de valor con costo de default más alto (Δ = 0.05)",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Deuda", 
yaxis_title = "Valor",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot7 = plot(scatters3, layout7)

savefig(plot7, "MuyOpcionalDelta.png")

#=========Mecánica de los shocks de preferencias============#
#Vamos a tratar de graficar las 3 funciones de valor, para cada χ, como función de la deuda

χ = [0.0001, 0.1]

unos = ones(200,1)

sd1.v
sd1.vD
sd1.vR
sd1.bgrid

sd1.ygrid[11]
sd1.v[:,11]

modsd1vD = sd1.vD[11] * unos
sd1.vR[:,11]

scatters1 = Vector{AbstractTrace}(undef, 3)
scatters1[1] = scatter(x = sd1.bgrid, y = sd1.v[:,11], name = "V",marker= attr(color="00FFFF"))
scatters1[2] = scatter(x = sd1.bgrid, y = modsd1vD[:,1], name = "vD",marker = attr(color="#8A2BE2"))
scatters1[3] = scatter(x = sd1.bgrid, y = sd1.vR[:,11], name = "vR",marker = attr(color = "#F8F1E9"))

layout4 = Layout(title="Funciones de valor con  χ = 0.1",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Deuda", 
yaxis_title = "Valor",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot4 = plot(scatters1, layout4)

savefig(plot4, "chichanges1.png")
#==================#
sd4 = Arellano(χ = 0.0001)
mpe!(sd4)

unos4 = ones(200,1)

sd4.v
sd4.vD
sd4.vR
sd4.bgrid

modsd4vD = sd4.vD[11] * unos4

scatters4 = Vector{AbstractTrace}(undef, 3)
scatters4[1] = scatter(x = sd4.bgrid, y = sd4.v[:,11], name = "V", marker= attr(color="00FFFF"))
scatters4[2] = scatter(x = sd4.bgrid, y = modsd4vD[:,1], name = "vD",marker = attr(color="#8A2BE2"))
scatters4[3] = scatter(x = sd4.bgrid, y = sd4.vR[:,11], name = "vR",marker = attr(color = "#F8F1E9"))

layout5 = Layout(title="Funciones de valor con χ = 0.0001",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Deuda", 
yaxis_title = "Valor",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot5 = plot(scatters4, layout5)

savefig(plot5, "chichanges2.png")

#=====Pruebita Doble=======#
prueba = Vector{AbstractTrace}(undef, 6)
prueba = Vector{AbstractTrace}(undef, 6)

# Scatter plot 1
prueba[1] = scatter(
    x = sd1.bgrid,
    y = sd1.v[:, 11],
    name = "V w/ χ = 0.1",
    marker = attr(color = "blue")  # Asigna un color azul
)

# Scatter plot 2
prueba[2] = scatter(
    x = sd1.bgrid,
    y = modsd1vD[:, 1],
    name = "vD w/ χ = 0.1",
    marker = attr(color = "purple")  # Asigna un color morado
)

# Scatter plot 3
prueba[3] = scatter(
    x = sd1.bgrid,
    y = sd1.vR[:, 11],
    name = "vR w/ χ = 0.1",
    marker = attr(color = "white")  # Asigna un color blanco
)

# Scatter plot 4
prueba[4] = scatter(
    x = sd4.bgrid,
    y = sd4.v[:, 11],
    name = "V w/ χ = 0.0001",
    marker = attr(color = "red")  # Asigna un color rojo
)

# Scatter plot 5
prueba[5] = scatter(
    x = sd4.bgrid,
    y = modsd4vD[:, 1],
    name = "vD w/ χ = 0.0001",
    marker = attr(color = "green")  # Asigna un color verde
)

# Scatter plot 6
prueba[6] = scatter(
    x = sd4.bgrid,
    y = sd4.vR[:, 11],
    name = "vR w/ χ = 0.0001",
    marker = attr(color = "orange")  # Asigna un color naranja
)

layout8 = Layout(title="Funciones de valor combinadas",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Deuda", 
yaxis_title = "Valor",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot8 = plot(prueba, layout8)

savefig(plot8, "valuefunctionsaltogether.png")
#============Cuántos defaults son por los precios?===============#





#=======Retorno a mercados según el ingreso======#
include("arellano1.4.jl")

sd5 = Arellano(ψ1 = 0)
mpe!(sd5)
sd6 = Arellano(ψ1 = 1)
mpe!(sd6)

scatters5 = Vector{AbstractTrace}(undef, 2)
scatters5[1] = scatter(x = sd5.bgrid, y = sd5.v[:,11], name = "V w/ ψ1 = 0", marker = attr(color = "red"))
scatters5[2] = scatter(x = sd6.bgrid, y = sd6.v[:,11], name = "V w/ ψ1 = 1",marker = attr(color = "blue"))

layout9 = Layout(title="Funciones de valor para el retorno de una economía defaulteada",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Deuda", 
yaxis_title = "Valor",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot9 = plot(scatters5, layout9)

savefig(plot9, "returntomarkets.png")

#======Retorno a mercados segun el monto de la deuda (Opcional)===========#
sd7 = Arellano(ψ1 = 0)
mpe!(sd7)
sd8 = Arellano(ψ1 = 1)
mpe!(sd8)

scatters6 = Vector{AbstractTrace}(undef, 2)
scatters6[1] = scatter(x = sd7.bgrid, y = sd7.v[:,11], name = "V w/ ψ1 = 0",marker = attr(color = "green"))
scatters6[2] = scatter(x = sd8.bgrid, y = sd8.v[:,11], name = "V w/ ψ1 = 1",marker = attr(color = "yellow"))

layout10 = Layout(title="Value functions for the return of the defaulted country second version",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Deuda", 
yaxis_title = "Valor",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot10 = plot(scatters6, layout10)

savefig(plot10, "optionalreturntomarkets.png")

prueba3 = Vector{AbstractTrace}(undef, 6)
prueba3[1] = scatter(x = sd7.bgrid, y = sd7.v[:,11], name = "V w/ χ = 0.1")
prueba3[2] = scatter(x = sd7.bgrid, y = sd7.vD[:,11], name = "vD w/ χ = 0.1")
prueba3[3] = scatter(x = sd7.bgrid, y = sd7.vR[:,11], name = "vR w/ χ = 0.1")
prueba3[4] = scatter(x = sd8.bgrid, y = sd8.v[:,11], name = "V w/ χ = 0.0001")
prueba3[5] = scatter(x = sd8.bgrid, y = sd8.vD[:,11], name = "vD w/ χ = 0.0001")
prueba3[6] = scatter(x = sd8.bgrid, y = sd8.vR[:,11], name = "vR w/ χ = 0.0001")

layoutqsy2 = Layout(title="Value functions in the return to default",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Debt", 
yaxis_title = "Value",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plotqsy2 = plot(prueba3, layoutqsy2)

layout11 = Layout(title="Repay and default zones with changes in the ψ function",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Income", 
yaxis_title = "Debt",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

repanddef131 = plot(contour(x = sd7.ygrid, y = sd7.bgrid, z = sd7.prob, colorbar=attr(title="Probability of default",titleside="right"), contours_coloring="heatmap", reversescale = true), layout11)
repanddef132 = plot(contour(x = sd8.ygrid, y = sd8.bgrid, z = sd8.prob, colorbar=attr(title="Probability of default",titleside="right"), contours_coloring="heatmap", reversescale = true), layout11)

savefig(repanddef131, "repayanddeftoptional1.png")
savefig(repanddef132, "repayanddeftoptional2.png")

#======Retorno a mercados segun el monto de la deuda (Opcional)===========#
sd30 = Arellano(ψ1 = 0)
mpe!(sd30)
sd31 = Arellano(ψ1 = 1)
mpe!(sd31)

scatters30 = Vector{AbstractTrace}(undef, 2)
scatters30[1] = scatter(x = sd30.bgrid, y = sd30.v[:,11], name = "V w/ ψ1 = 0")
scatters30[2] = scatter(x = sd31.bgrid, y = sd31.v[:,11], name = "V w/ ψ1 = 1")

layout30 = Layout(title="Value functions for the return of the defaulted country third version",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Debt", 
yaxis_title = "Value",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot30 = plot(scatters30, layout30)

prueba2 = Vector{AbstractTrace}(undef, 6)
prueba2[1] = scatter(x = sd30.bgrid, y = sd30.v[:,11], name = "V w/ χ = 0.1")
prueba2[2] = scatter(x = sd30.bgrid, y = sd30.vD[:,11], name = "vD w/ χ = 0.1")
prueba2[3] = scatter(x = sd30.bgrid, y = sd30.vR[:,11], name = "vR w/ χ = 0.1")
prueba2[4] = scatter(x = sd31.bgrid, y = sd31.v[:,11], name = "V w/ χ = 0.0001")
prueba2[5] = scatter(x = sd31.bgrid, y = sd31.vD[:,11], name = "vD w/ χ = 0.0001")
prueba2[6] = scatter(x = sd31.bgrid, y = sd31.vR[:,11], name = "vR w/ χ = 0.0001")

layoutqsy = Layout(title="Value functions in the return to default",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Debt", 
yaxis_title = "Value",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plotqsy = plot(prueba2, layoutqsy)

savefig(plot30, "optionaloftheoptionalsreturntomarkets.png")

layout31 = Layout(title="Repay and default zones with changes in the ψ function",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Income", 
yaxis_title = "Debt",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

repanddef141 = plot(contour(x = sd30.ygrid, y = sd30.bgrid, z = sd30.prob, colorbar=attr(title="Probability of default",titleside="right"), contours_coloring="heatmap", reversescale = true), layout31)
repanddef142 = plot(contour(x = sd31.ygrid, y = sd31.bgrid, z = sd31.prob, colorbar=attr(title="Probability of default",titleside="right"), contours_coloring="heatmap", reversescale = true), layout31)

savefig(repanddef141, "repayanddeftoptionaloftheoptionals1.png")
savefig(repanddef142, "repayanddeftoptionaloftheoptionals2.png")

#=====Simulador de Defaults======#

sd9 = Arellano(defcost_OG = 1)
mpe!(sd9)
bvec, cvec, yvec, dvec, qvec, pvec= simul(sd9)

debt_pbi = bvec./yvec

layout12 = Layout(title="Debt/PBI for 100 years",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Year", 
yaxis_title = "Debt/PBI",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot12 = plot(scatter(y = debt_pbi, opacity=0.75), layout12)

savefig(plot12, "debtoverpbi.png")

#--------------#

layout13 = Layout(title="Mean of the probability of default",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Year", 
yaxis_title = "%",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot13 = plot(scatter(y = dvec, opacity=0.75), layout13)

#Calculamos el porcentaje de Default
(mean(dvec)-1)*100

#====Simulación de distribución de defaults======#
N = zeros(1000)
for i in 1:1000
    ar = Arellano()
    mpe!(ar)
    bvec, cvec, yvec, dvec, qvec, pvec = simul(ar)
    N[i] = (mean(dvec)-1)*100
end


shapes = [vline(mean(N), line_dash="dot", line_color="#818181")]

annotations = [attr(x=mean(N), y=0, yanchor="top", yref="paper", showarrow=false, text="13.995")]

layout14 = Layout(shapes=shapes,
annotations=annotations,
title="Distribution of the default frequency with T = 1000",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="%", 
yaxis_title = "Years",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

dist_default = plot(histogram(x = N), layout14)

savefig(dist_default, "dist_default2.png")


#====Simulador con cambios en el costo de default======#
N = [i for i in range(0.01,0.2,20)]
pvec = similar(N)
pvec2 = similar(N)
T = similar(N)

#for i in 1:length(N)
#    for j in 1:length(N)
#        ar = Arellano(Δ = N[i])
#        mpe!(ar)
#        bvec, cvec, yvec, dvec, qvec, pvec = simul(ar)
#        pvec[j]
#    T[i] = (mean(dvec)-1)*100
#end

layout15 = Layout(title="Frequency of default with changes in Δ (cost of defaul)",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Year", 
yaxis_title = "%",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

plot_simul_Δ = plot(scatter(x = N, y = T), layout15)

savefig(plot_simul_Δ, "Plot_simul_Δ.png")


#=====Deuda al momento de default=======#
"""Intuitivamente, armo un for loop que cuando dvec sea 2 me diga bvec y pvec del período anterior"""

sd9 = Arellano(defcost_OG = 1)
mpe!(sd9)
bvec, cvec, yvec, dvec, qvec, pvec= simul(sd9)
debdef = zeros(100)
probdef = zeros(100)

for i in 1:length(dvec)
    if dvec[i] == 2
        debdef[i] = bvec[i-1]
        probdef[i] = pvec[i-1]
    end
end

sum(debdef)
sum(dvec)-100
sum(debdef)/(sum(dvec)-100)
sum(probdef)/(sum(dvec)-100)

#======Comparación con un modelo sin default======#
sd25 = Arellano(defcost_OG = 0, Δ = 0.9)
mpe!(sd25)
bvec, cvec, yvec, dvec, qvec, pvec= simul(sd25)

debt_pbi25 = bvec./yvec

layout25 = Layout(title="Debt/PBI for 100 years in the model without default and the model with default",
width=1920 * 0.5, height=1080 * 0.5,
legend=attr(orientation="h", x=0.02),
xaxis_title="Year", 
yaxis_title = "Debt/PBI",
xaxis=attr(zeroline=false, gridcolor="#434343"),
yaxis=attr(zeroline=false, gridcolor="#434343"),
paper_bgcolor="#272929", plot_bgcolor="#272929",
font_color="#F8F1E9", font_size=12,
font_family="Lato",
hovermode="x",
)

scatters25 = Vector{AbstractTrace}(undef, 2)
scatters25[1] = scatter(y = debt_pbi, opacity=0.75, name = "Model with default cost")
scatters25[2] = scatter(y = debt_pbi25, opacity=0.75, name = "Model with no default cost")

plot25 = plot(scatters25, layout25)

savefig(plot25, "Compdefandnodef.png")

#=========Deuda Larga===========#

include("deudalarga.jl")
include("def_simul.jl")


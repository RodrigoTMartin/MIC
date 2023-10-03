println("----- Vectores -----")

# Definimos un vector
vector = [1, 2, 3, 4, 5]

println("Vector original: ", vector)

for x in vector
    println("Elemento: ", x)
end

for x in vector
    println(x+2)
end

# Alguna operación muy básica
x = 2;
y = 4;
@show 2x - 3y;
@show x + y;
# El $ dentro de un string nos permite interpolar variables
"x + y = $(x + y)"

z = "x + y = $(x + y)"
typeof(z)

# Podemos concatenar strings con *
"Suma: "*z

# La forma más comun de acceder a los elementos de un vector es con el indice
print(vector[2])

# Los elementos de un vector se pueden cambiar
vector[2] = 56

# Pero siempre por objetos del tipo en el que fue definido
vector[2] = "Hola"

# Podemos crear un vector de distinto tipo con
vector2 = Any[1,2,3,4,5]
vector2[2] = "Hola"

# También se puede usar un convertor para opera en estas cosas
vector3 = [1, 2, 3, 4, 5]
vector3 = convert(Vector{Any}, vector3)
vector3[2] = "Hola"

# La funcion eachindex nos devuelve un iterador que nos permite acceder a los indices del vector
for x in eachindex(vector)
    println(x)
end

for x in enumerate(vector)
    println(x)
end

for x in vector
    println(x)
end

# Fijemonos las diferencias entre los dos for loops
for x in eachindex(vector)
    vector[x] = x+2
end

for x in eachindex(vector)
    vector[x] = vector[x]+2
end

# Ahora accedemos a un elemento del vector
for x in enumerate(vector)
    println("Elemento: ", x)
end

# Dado que enumerate nos devuelve una tupla, podemos desempaquetarla
for (i, x) in enumerate(vector)
    println("Elemento ", i, ": ", x)
end

# Sumamos un escalar a cada elemento (operación elemento a elemento)
vector = vector .+ 2

println("Vector luego de sumar 2 a cada elemento: ", vector)

println("\n----- Matrices -----")

# Definimos una matriz
matriz = [1 2 3; 4 5 6; 7 8 9]

println("Matriz original: ", matriz)

for x in eachindex(matriz)
    println(x)
end

for x in enumerate(matriz)
    println(x)
end

println(matriz[2])
println(matriz[2, 3])
println(matriz[:, 3])

# Multiplicamos cada elemento por un escalar (operación elemento a elemento)
matriz = matriz .* 2

println("Matriz luego de multiplicar cada elemento por 2: ", matriz)

# EJEMPLO: Defino una funcion de valor aleatoria en una matriz de 2x2
value = rand(30, 51)
# Si yo quiero trabajar iterando sobre todos los elementos de esta matriz, puedo acceder con un for loop
for (ia, jz) in enumerate(value)
    println("Elemento ", ia*2, ": ", jz*0.5)
end

ia = 2
jz = 3
println(value[ia, jz])

# Por lo general hacemos cosas un poquito más interesantes:
consumo = Matrix{Float64}(undef, 30, 51)
for (ia, jz) in enumerate(value)
    consumo[ia] = jz*0.5
end

println(consumo[1, 1])

println("\n----- Broadcasting -----")

# Definimos un vector y una matriz
vector = [1, 2, 3]
matriz = [1 2 3; 4 5 6; 7 8 9]

println("Vector: ", vector)
println("Matriz: ", matriz)

# Sumamos el vector a cada fila de la matriz (broadcasting)
resultado = matriz .+ vector #chequear bien esto

println("Resultado luego de sumar el vector a cada fila de la matriz: ", resultado)

# Al último elemento de los vectores (también en otras estructuras) se puede acceder con "end"
vector[end]

# Comprensiones
doubles = [2i for i in 1:0.5:4] # 2i = 2*i

println("----- Tuplas -----")

# Definimos una tupla
tupla = (1, 2, 3, 4, 5)

println("Tupla: ", tupla)

# Accedemos a un elemento de la tupla
elemento = tupla[1]

println("Primer elemento de la tupla: ", elemento)

# Intentamos cambiar un elemento de la tupla
try
    tupla[1] = 10
catch e
    println("Error al intentar cambiar un elemento de la tupla: ", e)
end

println("\n----- Tuplas y funciones -----")

# Definimos una función que devuelve una tupla
function min_max(vector)
    return (minimum(vector), maximum(vector))
end

vector = [1, 2, 3, 4, 5]
resultado = min_max(vector)

x = (1, 2, 3)


println("Mínimo y máximo del vector: ", resultado)

# Las tuplas son inmutables
x[1] = 10
try
    x[1] = 10
catch e
    println("Error al intentar cambiar un elemento de la tupla: ", e)
end

# Julia indexa a partir del 1!
x[0]
x[1]
typeof(x)

# Los elementos de una tupla también pueden ser de diferentes tipos
x = (1, "hola", 3.0)
typeof(x)

println("----- Diccionarios -----")

# Definimos un diccionario
diccionario = Dict("a" => 1, "b" => 2, "c" => 3)

println("Diccionario: ", diccionario)

# Accedemos a un valor del diccionario
valor = diccionario["a"]

println("Valor asociado a 'a': ", valor)

# Cambiamos un valor del diccionario
diccionario["a"] = 10

println("Diccionario luego de cambiar el valor asociado a 'a': ", diccionario)

# Añadimos un nuevo par clave-valor al diccionario
diccionario["d"] = 4

println("Diccionario luego de añadir el par 'd': 4: ", diccionario)

println("----- Sets -----")

# Definimos un set
conjunto = Set([1, 2, 3, 4, 5]);

println("Set: ", conjunto)

# Intentamos añadir un elemento que ya está en el set
push!(conjunto, 2);

println("Set luego de intentar añadir el 2: ", conjunto)

# Añadimos un nuevo elemento al set
push!(conjunto, 6);

println("Set luego de añadir el 6: ", conjunto)

println("\n----- Operaciones con sets -----")

# Definimos dos sets
conjuntoA = Set([1, 2, 3, 4, 5]);
conjuntoB = Set([4, 5, 6, 7, 8]);

println("Set A: ", conjuntoA)
println("Set B: ", conjuntoB)

# Calculamos la unión de los sets
unionAB = union(conjuntoA, conjuntoB);

println("Unión de A y B: ", unionAB)

# Calculamos la intersección de los sets
interseccionAB = intersect(conjuntoA, conjuntoB);

println("Intersección de A y B: ", intersect(conjuntoA, conjuntoB))

# Calculamos la diferencia de los sets
diferenciaAB = setdiff(conjuntoA, conjuntoB);

println("Diferencia de A y B (elementos en A pero no en B): ", diferenciaAB)

println("----- Structs -----")

# Definimos una estructura
struct Persona
    nombre::String
    edad::Int
end

# Creamos una instancia de la estructura
# Se puede pensar como que "corremos una función" que crea una instancia de la estructura
persona = Persona("Juan", 30);

println(persona)

# Accedemos a un campo de la estructura con el "." (dot syntax)
println("Nombre de la persona: ", persona.nombre)

# Intentamos cambiar un campo de la estructura
try
    persona.edad = 31
catch e
    println("Error al intentar cambiar un campo de la estructura: ", e)
end

println("----- Mutable Structs -----")

# Definimos una estructura mutable
mutable struct Empresa
    nombre::String
    capital::Float64
    empleados::Int
end

# Creamos una instancia de la estructura
empresa = Empresa("ACME Corp.", 100000.0, 50);

println("Empresa: ", empresa)

# Cambiamos un campo de la estructura
println("Empresa: ", empresa.capital)
empresa.capital -= 5000.0;
println("Empresa: ", empresa.capital)

println("Empresa luego de disminuir el capital en 5000: ", empresa)

# Añadimos empleados
empresa.empleados += 10

# Errores con estructuras y tipado
# Definimos una estructura con un campo de tipo Int64
struct MiEstructura
    campo1::Int64
end

# Creamos una instancia de la estructura
mi_instancia = MiEstructura(42);

println(mi_instancia.campo1)

# Intentamos asignar un valor de tipo incorrecto al campo
mi_instancia.campo1 = "hola"

# Definimos una estructura mutable con un campo de tipo Int64
mutable struct MiEstructura2
    campo1::Int64
end

# Creamos una instancia de la estructura
mi_instancia2 = MiEstructura2(42);

println(mi_instancia2.campo1)

# Ahora podemos cambiar el valor del campo1
mi_instancia2.campo1 = 56;
println(mi_instancia2.campo1)
# Pero si intentamos asignar un valor de tipo incorrecto al campo,
# Julia nos dará un error.
mi_instancia.campo1 = "hola"


#-------------------------------------------------
println("----- Tipado Dinámico -----")

# Asignamos un entero a una variable
x = 10

println("x es: ", x)

typeof(x)
# Reasignamos un string a la misma variable
x = "Hola Mundo"

println("Ahora x es: ", x)

typeof(x)
# Instalando paquetes
# using Pkg
# Pkg.add("DataFrames")

# Cargando paquetes
using DataFrames

println("----- DataFrames -----")

# Crear un DataFrame
df = DataFrame(nombre = ["Juan", "Ana", "Carlos"], edad = [25, 30, 35], ciudad = ["Madrid", "Barcelona", "Valencia"]);

println("DataFrame: ", df)

# Acceder a una columna
edades = df.edad

println("Edades: ", edades)

# Filtrar el DataFrame
df_filtrado = df[df.edad .> 25, :];

println("DataFrame filtrado: ", df_filtrado)

println("----- Tipos abstractos -----")

#-----------------------------------------------------------
# Ejemplo de tipos concretos y abstractos en Julia
# Crear una instancia de un tipo concreto Int (fijensé que es parecido a lo que hicimos antes con los structs)
x = Int(10);
println("El valor de x es: ", x)

# Crear una instancia de un tipo concreto Float64
y = Float64(20.0);
println("El valor de y es: ", y)

# Intenta crear una instancia de un tipo abstracto
try
    n = Number() # Esto dará un error
catch e
    println("Error al intentar instanciar un tipo abstracto: ", e)
end

# Ejemplo de uso de un tipo abstracto en una función
# Función que acepta un valor de cualquier tipo numérico
function addOne(num::Number)
    return num + 1
end

println("Resultado de addOne con un entero: ", addOne(x))
println("Resultado de addOne con un float: ", addOne(y))

# La función addOne funcionará con cualquier tipo de número, ya sea Int, Float64, Complex, etc.
z = Complex(2, 3)
println("Resultado de addOne con un número complejo: ", addOne(z))

#-----------------------------------------------------------
abstract type Agent end # No pueden tener campos!

struct Employed <: Agent
    wage::Float64
end

struct Unemployed <: Agent
    job_offer_rate::Float64 # Tasa de ofertas de trabajo: será la probabilidad de que reciba una oferta
    job_offer_wage::Float64 # Salario de la oferta de trabajo
end

function simulate_agent(agent::Agent, periods::Int)
    #La función toma dos argumentos: un agente (E o U) y un número de períodos para simular
    wages = Float64[] # Inicializamos un vector vacío
    # Es una buena práctica especificar el tipo de datos "Float64" en los contenedores porque ayuda a Julia a hacer optimizaciones de rendimiento.
    # Si no se especifica, que pasa? Nada, crea un vector de tipo "Any" que puede contener cualquier cosa
    for i in 1:periods
        if agent isa Employed # Con este chequeamos que tipo de agente es
            wage = agent.wage # Asignamos el salario del agente a la variable
            push!(wages, wage) # ! es una convención que para indicar que la función modificará o mutará al menos uno los argumentos de otro objeto.
            # Fancy way: push!(wages, agent.wage)
            # Introducimos la posibilidad de que un empleado pierda su trabajo
            # Con una probabilidad del 10%, el agente pierde su trabajo y pasa a ser desempleado.
            if rand() < 0.1 # rand() genera un número aleatorio entre 0 y 1
                agent = Unemployed(0.2, 100.0 + randn() * 10) 
                # El nuevo agente desempleado tiene una tasa de ofertas de trabajo del 20% 
                # Salario de oferta de trabajo que es 100 más o menos un ruido aleatorio
                # randn() genera un numero aleatorio de una distribución normal estándar
            end
        elseif agent isa Unemployed # Chequeamos si es desempleado
            if rand() < agent.job_offer_rate
                wage = agent.job_offer_wage
                # Si recibe una oferta de trabajo, su salario se convierte en el salario de la oferta de trabajo más un ruido aleatorio, y el agente pasa a estar empleado
                agent = Employed(wage + randn() * 10)
            else
                wage = 10.0 # Subsidio por desempleo
            end
            push!(wages, wage)
        end
    end
    
    return wages
end

# Simulamos a un agente empleado con un salario de 100 durante 400 periodos
employed_agent = Employed(100.0)
employed_wages = simulate_agent(employed_agent, 400)

# Simulamos a un agente desempleado con una tasa de ofertas de trabajo de 0.2 y un salario de oferta de trabajo de 100 durante 400 periodos
unemployed_agent = Unemployed(0.2, 100.0)
unemployed_wages = simulate_agent(unemployed_agent, 400)

using PlotlyJS

function plot_simulation(employed_wages, unemployed_wages)
    trace1 = scatter(y=employed_wages, mode="lines", name="Empleado")
    trace2 = scatter(y=unemployed_wages, mode="lines", name="Desempleado")
    layout = Layout(title="Simulación de salarios de agentes",
                    xaxis=attr(title="Periodos"),
                    yaxis=attr(title="Salario"))
    plot([trace1, trace2], layout)
end

plot_simulation(employed_wages, unemployed_wages)

#-----------------------------------------------------------
# Graficos con PlotlyJS
tr = scatter(x=1:4, y=[10, 11, 12, 13], mode="lines")
plot(tr)

# Funciones

function suma(x, y)
    return x + y
end

suma(3, 4)

function describe_number(x::Int)
    if x > 0
        println("El número es positivo.")
    elseif x < 0
        println("El número es negativo.")
    else
        println("El número es cero.")
    end

    if x % 2 == 0
        println("El número es par.")
    else
        println("El número es impar.")
    end
end

# Multiple dispatch y tipado dinamico también nos permiten generar otras funciones con el mismo nombre
function describe_number(x::Float64)
    x = round(x)
    if x > 0
        println("El número es positivo.")
    elseif x < 0
        println("El número es negativo.")
    else
        println("El número es cero.")
    end

    if x % 2 == 0
        println("El número es par.")
    else
        println("El número es impar.")
    end
end

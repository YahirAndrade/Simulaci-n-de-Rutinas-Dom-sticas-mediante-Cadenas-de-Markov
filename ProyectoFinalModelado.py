import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np
from matplotlib.animation import FuncAnimation

def obtener_horario(minuto):
    hora = minuto // 60
    if 0 <= hora <= 6:
        return "Madrugada"
    elif 6 < hora <= 12:
        return "Mañana"
    elif 12 < hora <= 18:
        return "Tarde"
    else:
        return "Noche"

def simular_dia_montecarlo_con_horarios(matrices_por_horario, transiciones_por_horario, estado_inicial, pasos, tasa_actualizacion):
    trayectoria = []
    estado_actual = estado_inicial
    actividad_actual = actividad_en_habitacion(estado_actual)
    duracion_restante = sample_activity_duration(actividad_actual) // tasa_actualizacion

    for i in range(pasos):
        minuto = i * tasa_actualizacion
        horario = obtener_horario(minuto)

        # Forzar comidas obligatorias
        hora = minuto // 60
        if 7 <= hora < 9 and estado_actual == "Cocina":  # Desayuno
            actividad_actual = "Comer"
            duracion_restante = sample_activity_duration("Comer") // tasa_actualizacion - 1
        elif 13 <= hora < 15 and estado_actual == "Cocina":  # Comida
            actividad_actual = "Comer"
            duracion_restante = sample_activity_duration("Comer") // tasa_actualizacion - 1
        elif 19 <= hora < 21 and estado_actual == "Cocina":  # Cena
            actividad_actual = "Comer"
            duracion_restante = sample_activity_duration("Comer") // tasa_actualizacion - 1

        # Forzar sueño nocturno
        if 22 <= hora < 24 or 0 <= hora < 6 and estado_actual == "Recámara":
            actividad_actual = "Dormir"
            duracion_restante = sample_activity_duration("Dormir") // tasa_actualizacion - 1

        trayectoria.append((minuto, estado_actual, actividad_actual))

        if duracion_restante > 0:
            duracion_restante -= 1
            if actividad_actual == "Salir de vacaciones" and estado_actual == "Fuera de casa":
                continue
        else:
            sub_probs = matrices_por_horario[horario][estado_actual]
            sub_estados = sub_actividades[estado_actual]
            actividad_actual = random.choices(sub_estados, weights=sub_probs, k=1)[0]
            duracion_restante = sample_activity_duration(actividad_actual) // tasa_actualizacion - 1

        opciones = transiciones[estado_actual]
        probs = transiciones_por_horario[horario][estado_actual]
        if actividad_actual == "Salir de vacaciones" and estado_actual == "Fuera de casa":
            estado_actual = "Fuera de casa"
        else:
            estado_actual = random.choices(opciones, weights=probs, k=1)[0]

    return trayectoria

def actividad_en_habitacion(habitacion):
    actividades = sub_actividades[habitacion]
    probs = sub_probabilidades[habitacion]
    return random.choices(actividades, weights=probs, k=1)[0]

def sample_activity_duration(actividad):
    activity_durations = {
        "Preparar comida": (15, 30), "Comer": (20, 40), "Lavar trastes": (10, 20),
        "Ver la TV": (30, 90), "Dormir": (360, 480), "Leer": (20, 60), "Platicar": (30, 90),
        "Hacer del baño": (5, 10), "Bañar": (15, 30), "Arreglarse": (10, 20),
        "Reposar": (20, 60), "Rezar": (5, 15),
        "Regar las plantas": (10, 20), "Tomar el aire": (20, 60),
        "Salir de compras": (60, 120), "Visitar familia": (120, 240), "Salir de vacaciones": (2880, 10080)
    }
    # Ajuste para siestas en Sala
    if actividad == "Dormir" and sub_actividades.get("Sala", []):
        return random.randint(60, 120)  # 1–2 h para siestas
    min_dur, max_dur = activity_durations[actividad]
    return random.randint(min_dur, max_dur)

# Crear espacio físico
espacio_fisico = ["Cocina", "Sala", "Baño", "Recámara", "Patio", "Fuera de casa"]

# Conexiones entre el espacio físico
conexiones = [
    ("Sala", "Sala"), ("Cocina", "Cocina"), ("Recámara", "Recámara"),
    ("Baño", "Baño"), ("Patio", "Patio"), ("Fuera de casa", "Fuera de casa"),
    ("Cocina", "Sala"), ("Sala", "Cocina"), ("Sala", "Recámara"),
    ("Recámara", "Sala"), ("Sala", "Baño"), ("Baño", "Sala"),
    ("Patio", "Sala"), ("Sala", "Patio"), ("Patio", "Fuera de casa"),
    ("Fuera de casa", "Patio")
]

# Crear grafo con dirección
grafo = nx.DiGraph()
grafo.add_nodes_from(espacio_fisico)
grafo.add_edges_from(conexiones)

# Mostrar grafo estático
plt.figure(figsize=(6, 4))
nx.draw(grafo, with_labels=True, node_color="lightgreen", node_size=1300, arrowstyle="->")
plt.title("Distribución del espacio físico")
plt.show()
plt.close()

# Configuración de la simulación
tasa_actualizacion = 5
num_actualizaciones = int(1440 / tasa_actualizacion)

transiciones = {
    "Cocina": ["Cocina", "Sala"],
    "Sala": ["Sala", "Cocina", "Baño", "Recámara", "Patio"],
    "Baño": ["Baño", "Sala"],
    "Recámara": ["Recámara", "Sala"],
    "Patio": ["Sala", "Fuera de casa", "Patio"],
    "Fuera de casa": ["Patio", "Fuera de casa"]
}

# Probabilidades de transición por horario
transiciones_por_horario = {
    "Madrugada": {
        "Cocina": [0.2, 0.8],  # Beber agua o regresar
        "Sala": [0.3, 0.2, 0.3, 0.2, 0.0],  # Posible baño
        "Baño": [0.4, 0.6],  # Hacer del baño
        "Recámara": [0.9, 0.1],  # Dormir
        "Patio": [0.9, 0.0, 0.1],  # Muy raro
        "Fuera de casa": [0.95, 0.05]
    },
    "Mañana": {
        "Cocina": [0.3, 0.7],  # Desayunar
        "Sala": [0.4, 0.2, 0.2, 0.1, 0.1],  # Activo
        "Baño": [0.2, 0.8],  # Bañarse
        "Recámara": [0.4, 0.6],  # Prepararse
        "Patio": [0.4, 0.4, 0.2],  # Salir
        "Fuera de casa": [0.3, 0.7]
    },
    "Tarde": {
        "Cocina": [0.3, 0.7],  # Comer
        "Sala": [0.4, 0.2, 0.2, 0.1, 0.1],  # Ver TV
        "Baño": [0.3, 0.7],  # Usar baño
        "Recámara": [0.6, 0.4],  # Siesta
        "Patio": [0.4, 0.4, 0.2],  # Salir
        "Fuera de casa": [0.3, 0.7]
    },
    "Noche": {
        "Cocina": [0.4, 0.6],  # Cenar
        "Sala": [0.5, 0.2, 0.2, 0.1, 0.0],  # Ver TV
        "Baño": [0.3, 0.7],  # Prepararse
        "Recámara": [0.9, 0.1],  # Dormir
        "Patio": [0.8, 0.1, 0.1],  # Raro
        "Fuera de casa": [0.9, 0.1]
    }
}

sub_actividades = {
    "Cocina": ["Preparar comida", "Comer", "Lavar trastes"],
    "Sala": ["Ver la TV", "Dormir", "Leer", "Platicar"],
    "Baño": ["Hacer del baño", "Bañar", "Arreglarse"],
    "Recámara": ["Dormir", "Reposar", "Rezar", "Leer"],
    "Patio": ["Regar las plantas", "Tomar el aire", "Platicar"],
    "Fuera de casa": ["Salir de compras", "Visitar familia", "Salir de vacaciones"]
}

sub_probabilidades = {
    "Cocina": [0.30, 0.50, 0.20],
    "Sala": [0.50, 0.20, 0.15, 0.15],
    "Baño": [0.50, 0.35, 0.15],
    "Recámara": [0.60, 0.20, 0.10, 0.10],
    "Patio": [0.40, 0.40, 0.20],
    "Fuera de casa": [0.78, 0.20, 0.02]
}

matrices_por_horario = {
    "Madrugada": {
        "Recámara": [0.85, 0.07, 0.05, 0.03],  # Dormir
        "Baño": [0.70, 0.20, 0.10],  # Hacer del baño
        "Cocina": [0.20, 0.60, 0.20],  # Beber agua
        "Sala": [0.60, 0.15, 0.15, 0.10],  # Menos dormir
        "Fuera de casa": [0.99, 0.01, 0.00],
        "Patio": [0.10, 0.70, 0.20]
    },
    "Mañana": {
        "Recámara": [0.10, 0.40, 0.20, 0.30],  # Prepararse
        "Baño": [0.20, 0.60, 0.20],  # Bañarse
        "Cocina": [0.30, 0.60, 0.10],  # Desayuno
        "Sala": [0.50, 0.00, 0.30, 0.20],  # Sin dormir
        "Fuera de casa": [0.78, 0.20, 0.02],
        "Patio": [0.30, 0.50, 0.20]
    },
    "Tarde": {
        "Recámara": [0.20, 0.40, 0.20, 0.20],  # Siesta
        "Baño": [0.50, 0.30, 0.20],  # Usar baño
        "Cocina": [0.30, 0.60, 0.10],  # Comida
        "Sala": [0.50, 0.20, 0.15, 0.15],  # Siesta posible
        "Fuera de casa": [0.78, 0.20, 0.02],
        "Patio": [0.30, 0.50, 0.20]
    },
    "Noche": {
        "Recámara": [0.85, 0.05, 0.05, 0.05],  # Dormir
        "Baño": [0.50, 0.30, 0.20],  # Prepararse
        "Cocina": [0.30, 0.60, 0.10],  # Cena
        "Sala": [0.60, 0.00, 0.20, 0.20],  # Sin dormir
        "Fuera de casa": [0.90, 0.10, 0.00],
        "Patio": [0.10, 0.70, 0.20]
    }
}

# Simulación
n_simulaciones = 365
resultados = []

for _ in range(n_simulaciones):
    trayectoria_simulada = simular_dia_montecarlo_con_horarios(
        matrices_por_horario, transiciones_por_horario, "Recámara", num_actualizaciones, tasa_actualizacion
    )
    resultados.append(trayectoria_simulada)

# Contar visitas y comidas
visitas_fuera = [sum(1 for x in tray if x[1] == "Fuera de casa") for tray in resultados]
vacaciones = [sum(1 for x in tray if x[2] == "Salir de vacaciones") for tray in resultados]
comidas = [sum(1 for x in tray if x[2] == "Comer" and 7 <= (x[0] // 60) < 9 or 13 <= (x[0] // 60) < 15 or 19 <= (x[0] // 60) < 21) for tray in resultados]
promedio_visitas_fuera = np.mean(visitas_fuera)
promedio_vacaciones = np.mean(vacaciones)
promedio_comidas = np.mean(comidas)
print(f"Promedio de visitas a 'Fuera de casa' por día: {promedio_visitas_fuera:.2f} (de {num_actualizaciones} pasos)")
print(f"Promedio de pasos en 'Salir de vacaciones' por día: {promedio_vacaciones:.2f}")
print(f"Promedio de comidas por día: {promedio_comidas:.2f}")

# Visualización de resultados por habitación
conteos = []
for tray in resultados:
    habitaciones = [x[1] for x in tray]
    conteo = Counter(habitaciones)
    conteos.append([conteo[estado] for estado in espacio_fisico])

conteos_array = np.array(conteos)
promedios = conteos_array.mean(axis=0)

plt.figure(figsize=(8, 4))
sns.barplot(x=espacio_fisico, y=promedios, hue=espacio_fisico, palette="mako", legend=False)
plt.title("Tiempo promedio en cada habitación por día (simulación Monte Carlo)")
plt.ylabel("Número de pasos (5 min cada uno)")
plt.xlabel("Habitación")
plt.show()

# Tabla horaria de actividades
df_actividades = pd.DataFrame(resultados[0], columns=["Minuto", "Habitación", "Actividad"])
actividades_unicas = df_actividades["Actividad"].unique()
tabla_actividades = pd.DataFrame(0, index=range(24), columns=actividades_unicas)

for _, fila in df_actividades.iterrows():
    hora = fila["Minuto"] // 60
    actividad = fila["Actividad"]
    if hora < 24:
        tabla_actividades.loc[hora, actividad] += 1

tabla_actividades_prop = tabla_actividades.div(tabla_actividades.sum(axis=1), axis=0).fillna(0)

plt.figure(figsize=(14, 6))
sns.heatmap(tabla_actividades_prop, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Distribución horaria de actividades (proporciones)")
plt.xlabel("Actividad")
plt.ylabel("Hora del día")
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.show()

# Imprimir períodos para minutos específicos
for minuto in [0, 301, 721, 1081]:
    print(f"Minuto {minuto}: {obtener_horario(minuto)}")

# Animación de un día
trayectoria = resultados[0]
pos = {
    "Cocina": (0, 0), "Sala": (1, 0), "Baño": (1, 1),
    "Recámara": (0, 1), "Patio": (2, 0), "Fuera de casa": (3, 0)
}

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xticks([])
ax.set_yticks([])

def update(frame):
    ax.clear()
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    minuto, habitacion, actividad = trayectoria[frame]
    hora = minuto // 60
    min_str = minuto % 60
    tiempo = f"{hora:02d}:{min_str:02d}"
    horario = obtener_horario(minuto)

    node_colors = ["lightgreen" if node != habitacion else "red" for node in grafo.nodes()]
    nx.draw(grafo, pos, ax=ax, with_labels=True, node_color=node_colors, node_size=1300, arrowstyle="->")

    info = f"Hora: {tiempo}\nPeríodo: {horario}\nHabitación: {habitacion}\nActividad: {actividad}"
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    progress = minuto / 1440
    ax.barh(-0.3, progress * 3.5, height=0.1, left=-0.5, color='blue', alpha=0.5)
    ax.text(-0.5, -0.35, "Progreso del día", fontsize=8)

    plt.title("Simulación de un día")

ani = FuncAnimation(fig, update, frames=len(trayectoria), interval=20, repeat=False)
plt.show()
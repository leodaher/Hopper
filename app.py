from collections import defaultdict
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import numpy as np
from random import randint
from flask import Flask, request, jsonify
from flask_cors import CORS

# Iniciando o Flask
app = Flask(__name__)
CORS(app)

# Lendo dados do arquivo gerado para o problema 1
df = pd.read_csv("cidades.csv")
df = df.head(50)
lat = df["latitude"]
long = df["longitude"]

# Inicializa matrizes com zeros
dist_arestas = np.zeros((len(lat), len(lat)))
fi_arestas = np.zeros((len(lat), len(lat)))

# Inicializa vetores com origens e destinos dos produtos
src = []
dest = []

# Funcao que calcula a distancia entre duas cidades dadas as coordenadas
def dist(lat1, lon1, lat2, lon2):
    R = 6373.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Insere a distancia entre cada cidade na matriz
for i in range(len(lat)):
    for j in range(len(lat)):
        dist_arestas[i][j] = dist(lat[i], long[i], lat[j], long[j])
        fi_arestas[i][j] = 1

custo_arestas = dist_arestas

# Classe utilizada para representar um grafo
class Graph:
    def minDistance(self, dist, queue):
        minimum = float("Inf")
        min_index = -1

        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index

    def printPath(self, parent, j, path):
        if parent[j] == -1 :
            path.append(j)
            return path
        self.printPath(parent , parent[j], path)
        path.append(j)

    def printSolution(self, dist, parent, src, dest):
        path = []
        self.printPath(parent, dest, path)
        return path

    def dijkstra(self, graph, src, dest):
        row = len(graph)
        col = len(graph[0])
        dist = [float("Inf")] * row
        parent = [-1] * row
        dist[src] = 0
        queue = []

        for i in range(row):
            queue.append(i)

        while queue:
            u = self.minDistance(dist,queue)
            queue.remove(u)

            for i in range(col):
                if graph[u][i] and i in queue:
                    if dist[u] + graph[u][i] < dist[i]:
                        dist[i] = dist[u] + graph[u][i]
                        parent[i] = u

        path = self.printSolution(dist, parent, src, dest)
        return path

# Funcao para o valor de uma aresta dada a distancia entre vertices e fluxo
def valor_aresta(d, fi):
    return 1.178175 * d / fi

# Funcao que recalcula todas as arestas
def calcula_arestas():
    for i in range(len(custo_arestas)):
        for j in range(len(custo_arestas[0])):
            custo_arestas[i][j] = valor_aresta(
                dist_arestas[i][j], fi_arestas[i][j])

# Retorna a matriz de fluxos dado um caminho
def calcula_fi(path):
    length = len(path)
    for i in range(length - 1):
        fi_arestas[path[i]][path[i+1]] += 1
        fi_arestas[path[i+1]][path[i]] += 1

# Funcao que converte lista de indices para lista de coordenadas
def convert_to_lat_long(path):
    result = []
    for i in range(len(path)):
        latCoord = lat[path[i]]
        longCoord = long[path[i]]
        result.append((latCoord, longCoord))
    return result

# Funcao para calcular melhores rotas dado numero de produtos
def calculate_routes(N):
    rotas = []
    # Funcao que gera lista aleatoria de produtos com origem e destino (seria fornecido diariamente pela Loggi)
    for i in range(N):
        src.append(randint(0, len(dist_arestas) - 1))

        # Verifica se origem == destino
        destRand = randint(0, len(dist_arestas) - 1)
        while(destRand == src[i]):
            destRand = randint(0, len(dist_arestas) - 1)

        dest.append(destRand)

    g = Graph()

    # Quao maior o valor utilizado no iterador maior a precisao do resultado
    for i in range(10):
        # Recalcula arestas
        calcula_arestas()

        # Para cada produto calcula o caminho mais barato
        for j in range(N):
            path = g.dijkstra(custo_arestas, src[j], dest[j])
            result = convert_to_lat_long(path)
            calcula_fi(path)

            if(i == 9):
                rotas.append(result)

    return rotas

@app.route('/routing/<int:quant>', methods=['GET'])
def routing(quant):
    rotas = calculate_routes(quant)
    return jsonify(rotas)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)

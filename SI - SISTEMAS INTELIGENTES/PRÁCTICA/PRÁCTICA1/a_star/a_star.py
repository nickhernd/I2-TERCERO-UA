import math
from queue import PriorityQueue
from typing import List, Tuple
from casilla import Casilla
from mapa import Mapa

class a_star:
    def __init__(self, mapa: Mapa, origen: Casilla, destino: Casilla):
        self.mapa = mapa
        self.origen = origen
        self.destino = destino
        self.n = mapa.getAlto()
        self.m = mapa.getAncho()
        self.dx = [-1, -1, 0, 1, 1, 1, 0, -1]
        self.dy = [0, 1, 1, 1, 0, -1, -1, -1]
        self.moves_map = [
            [8, 1, 2],
            [7, 0, 3],
            [6, 5, 4]
        ]
        self.best_cost = float('inf')
        self.best_moves = []

    def calcular_coste_movimiento(self, x1: int, y1: int, x2: int, y2: int) -> float:
        if x1 == x2 or y1 == y2:  # Movimiento horizontal o vertical
            return 1.0
        else:  # Movimiento diagonal
            return 1.5

    def heuristica_manhattan(self, x: int, y: int) -> float:
        return abs(x - self.destino.getFila()) + abs(y - self.destino.getCol())

    def heuristica_euclidea(self, x: int, y: int) -> float:
        return math.sqrt((x - self.destino.getFila())**2 + (y - self.destino.getCol())**2)

    def heuristica_cero(self, x: int, y: int) -> float:
        return 0

    def generar_hijos(self, node: Tuple[int, int, float, float, List[int]], pq: PriorityQueue, visited: List[List[bool]], g_costs: List[List[float]], heuristica):
        x, y, g, _, path = node

        for i in range(8):
            nx, ny = x + self.dx[i], y + self.dy[i]
            if 0 <= nx < self.n and 0 <= ny < self.m and self.mapa.getCelda(nx, ny) != 1:  # No es un muro
                new_g = g + self.calcular_coste_movimiento(x, y, nx, ny)
                if not visited[nx][ny] or new_g < g_costs[nx][ny]:
                    h = heuristica(nx, ny)
                    f = new_g + h
                    new_path = path + [self.moves_map[self.dx[i] + 1][self.dy[i] + 1]]
                    pq.put((f, (nx, ny, new_g, h, new_path)))
                    g_costs[nx][ny] = new_g

    def buscar(self, heuristica=None) -> Tuple[float, List[List[str]]]:
        if heuristica is None:
            heuristica = self.heuristica_cero

        pq = PriorityQueue()
        visited = [[False for _ in range(self.m)] for _ in range(self.n)]
        g_costs = [[float('inf') for _ in range(self.m)] for _ in range(self.n)]
        camino = [['.'] * self.m for _ in range(self.n)]

        start_h = heuristica(self.origen.getFila(), self.origen.getCol())
        pq.put((start_h, (self.origen.getFila(), self.origen.getCol(), 0, start_h, [])))
        g_costs[self.origen.getFila()][self.origen.getCol()] = 0

        while not pq.empty():
            _, (x, y, g, h, path) = pq.get()

            if x == self.destino.getFila() and y == self.destino.getCol():
                self.best_cost = g
                self.best_moves = path
                break

            if visited[x][y]:
                continue

            visited[x][y] = True
            self.generar_hijos((x, y, g, h, path), pq, visited, g_costs, heuristica)

        if self.best_cost != float('inf'):
            x, y = self.origen.getFila(), self.origen.getCol()
            camino[x][y] = 'x'
            for move in self.best_moves:
                if move == 1:
                    x -= 1
                elif move == 2:
                    x -= 1
                    y += 1
                elif move == 3:
                    y += 1
                elif move == 4:
                    x += 1
                    y += 1
                elif move == 5:
                    x += 1
                elif move == 6:
                    x += 1
                    y -= 1
                elif move == 7:
                    y -= 1
                elif move == 8:
                    x -= 1
                    y -= 1

                camino[x][y] = 'x'

        return self.best_cost, camino

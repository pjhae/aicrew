
# 이 코드를 사용할까 했는데 일단 이미지로 저장된 장매물 데이터를 wall(선분)로 만드는 것이 큰 난관이다.
# 0~360도 주변을 수십 수백 개의 방향으로 나누어 각 방향별로 충돌을 체트하는 코드이다.
# 이러한 방식은 기존의 bresenham을 이용해도 비슷한 결과를 만들 수 있다.
# 그래서 agent의 가시거리를 반경으로 하는 원을 mask base로 두고 0~360도를 일정 수 만큼 나누어 주변 장애물과 충돌을 bresenham 알고리즘으로
# 연산한다. 장애물 넘어의 공간은 maks 0, 장애물 까지의 거리는 mask 1로 지정한다. 이것을 agent의 교전 observation으로 활용한다.

from pygame.locals import *
from operator import add, sub
import pygame의
import sys
import math
import random

pygame.init()

# -----Options-----
WINDOW_SIZE = (1200, 800)  # Width x Height in pixels
NUM_RAYS = 150  # Must be between 1 and 360
SOLID_RAYS = False  # Can be somewhat glitchy. For best results, set NUM_RAYS to 360
NUM_WALLS = 5  # The amount of randomly generated walls
# ------------------

screen = pygame.display.set_mode(WINDOW_SIZE)
display = pygame.Surface(WINDOW_SIZE)

mx, my = pygame.mouse.get_pos()
lastClosestPoint = (0, 0)
running = True
rays = []
walls = []
particles = []


class Ray:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.dir = (math.cos(angle), math.sin(angle))

    def update(self, mx, my):
        self.x = mx
        self.y = my

    def checkCollision(self, wall):
        x1 = wall.start_pos[0]
        y1 = wall.start_pos[1]
        x2 = wall.end_pos[0]
        y2 = wall.end_pos[1]

        x3 = self.x
        y3 = self.y
        x4 = self.x + self.dir[0]
        y4 = self.y + self.dir[1]

        # Using line-line intersection formula to get intersection point of ray and wall
        # Where (x1, y1), (x2, y2) are the ray pos and (x3, y3), (x4, y4) are the wall pos
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        if denominator == 0:
            return None

        t = numerator / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 1 > t > 0 and u > 0:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            collidePos = [x, y]
            return collidePos


class Wall:
    def __init__(self, start_pos, end_pos, color='white'):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
        self.slope_x = end_pos[0] - start_pos[0]
        self.slope_y = end_pos[1] - start_pos[1]
        if self.slope_x == 0:
            self.slope = 0
        else:
            self.slope = self.slope_y / self.slope_x
        self.length = math.sqrt(self.slope_x ** 2 + self.slope_y ** 2)

    def draw(self):
        pygame.draw.line(display, self.color, self.start_pos, self.end_pos, 3)


for i in range(0, 360, int(360 / NUM_RAYS)):
    rays.append(Ray(mx, my, math.radians(i)))


def drawRays(rays, walls, color='white'):
    global lastClosestPoint
    for ray in rays:
        closest = 100000
        closestPoint = None
        for wall in walls:
            intersectPoint = ray.checkCollision(wall)
            if intersectPoint is not None:
                # Get distance between ray source and intersect point
                ray_dx = ray.x - intersectPoint[0]
                ray_dy = ray.y - intersectPoint[1]
                # If the intersect point is closer than the previous closest intersect point, it becomes the closest intersect point
                distance = math.sqrt(ray_dx ** 2 + ray_dy ** 2)
                if (distance < closest):
                    closest = distance
                    closestPoint = intersectPoint

        if closestPoint is not None:
            pygame.draw.line(display, color, (ray.x, ray.y), closestPoint)
            if SOLID_RAYS:
                pygame.draw.polygon(display, color, [(mx, my), closestPoint, lastClosestPoint])
                lastClosestPoint = closestPoint


def generateWalls():
    walls.clear()

    walls.append(Wall((0, 0), (WINDOW_SIZE[0], 0)))
    walls.append(Wall((0, 0), (0, WINDOW_SIZE[1])))
    walls.append(Wall((WINDOW_SIZE[0], 0), (WINDOW_SIZE[0], WINDOW_SIZE[1])))
    walls.append(Wall((0, WINDOW_SIZE[1]), (WINDOW_SIZE[0], WINDOW_SIZE[1])))

    for i in range(NUM_WALLS):
        start_x = random.randint(0, WINDOW_SIZE[0])
        start_y = random.randint(0, WINDOW_SIZE[1])
        end_x = random.randint(0, WINDOW_SIZE[0])
        end_y = random.randint(0, WINDOW_SIZE[1])
        walls.append(Wall((start_x, start_y), (end_x, end_y)))


def draw():
    display.fill((0, 0, 0))

    for wall in walls:
        wall.draw()

    for particle in particles:
        particle.draw()

    drawRays([ray for ray in rays], [wall for wall in walls])

    screen.blit(display, (0, 0))

    pygame.display.update()


generateWalls()
# while running:
#     mx, my = pygame.mouse.get_pos()
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             sys.exit()
#             pygame.quit()
#
#         if event.type == KEYDOWN:
#             # Re-randomize walls on Space
#             if event.key == pygame.K_SPACE:
#                 generateWalls()
#
#     for ray in rays:
#         ray.update(mx, my)
#
#     draw()
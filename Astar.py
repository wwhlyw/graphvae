import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_hexagon(x_center, y_center, size, color='yellow'):
    """绘制一个六边形"""
    angle = np.linspace(1/6*np.pi, 2*np.pi+1/6*np.pi, 7)  # 生成六边形的六个顶点
    x = x_center + size * np.cos(angle)
    y = y_center + size * np.sin(angle)
    plt.fill(x, y, color=color , edgecolor='black', linewidth=1)

def plot_hive(rows, cols, size, path, blocks):
    """绘制蜂巢"""

    for i in range(rows):
        for j in range(cols):
            x = - (i % 2) * np.sqrt(3) / 2 + j * size * np.sqrt(3)
            y = i * size * 1.5
            if i*100+j in blocks:
                plot_hexagon(x, y, size, color='red')
            elif i * 100 + j in path:
                plot_hexagon(x, y, size, color='blue')
            else:
                plot_hexagon(x, y, size, color='yellow')



from typing import List, Optional

def a_star_search(start, end, maps):
    open_list = []
    close_list = []
    open_list.append(start)
    while len(open_list) > 0:
        current_grid = find_min_gird(open_list)
        
        open_list.remove(current_grid)
        close_list.append(current_grid)
        neighbors = find_neighbors(current_grid, maps, open_list, close_list)
        for grid in neighbors:
            if grid not in open_list:
                grid.init_grid(current_grid, end)
                open_list.append(grid)
        for grid in open_list:
            if (grid.x == end.x) and (grid.y == end.y):
                return grid
    return None


def find_min_gird(open_list=[]):
    temp_grid = open_list[0]
    for grid in open_list:
        if grid.f < temp_grid.f:
            temp_grid = grid
    return temp_grid


def find_neighbors(grid, maps, open_list=[], close_list=[]):
    grid_list = []
    for grid in grid.neighbors:
        grid = Grid(grid // 100, grid % 100, maps)
        if is_valid_grid(grid, open_list, close_list):
            grid_list.append(grid)
 
    return grid_list


def is_valid_grid(grid, open_list=[], close_list=[]):

    # 是否已经在open_list中
    if contain_grid(open_list, grid):
        return False
    # 是否已经在closeList中
    if contain_grid(close_list, grid):
        return False
    return True


def contain_grid(open_list, grid):
    for open_grid in open_list:
        if (open_grid.x == grid.x) and (open_grid.y == grid.y):
            return True
    return False        
    
class Grid:
    def __init__(self, x, y, map, col= 77, row=92):
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.map = map
        self.row = row

        self.neighbors = self.map[self.x * col + self.y]
        self.parent = None


    def init_grid(self, parent, end):
        self.parent = parent
        if parent is not None:
            self.g = parent.g + 1
        else:
            self.g = 1
        self.h = abs(self.x - end.x) + abs(self.y - end.y)
        self.f = self.g + self.h
    



# 绘制地图
direction_1 = [[0, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]]
direction_0 = [[0, 1], [-1, 1], [-1, 0], [0, -1], [1, 0], [1, 1]]
def map1(row, col, blocks):
    graph = []
    tmp_graph = []
    for i in range(row):
        tmp = []
        for j in range(col):
            tmp.append(i * 100 + j)
        tmp_graph.append(tmp)
    for i in range(row):
        for j in range(col):
            tmp = []
            if i * 100 + j in blocks:
                graph.append(tmp)
                continue
            if i % 2:
                for k in direction_1:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if new_i >= 0 and new_i < row and new_j >= 0 and new_j < col and (new_i*100+new_j) not in blocks:
                        tmp.append(tmp_graph[new_i][new_j])
            else:
                for k in direction_0:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if new_i >= 0 and new_i < row and new_j >= 0 and new_j < col and (new_i*100+new_j) not in blocks:
                        tmp.append(tmp_graph[new_i][new_j])
            graph.append(tmp)
   
    return graph


def map(blocks):
    map_datas = pickle.load(open('./maps/map_53/cost.pickle', 'rb'))
    maps = []
    for i in range(92):
        for j in range(77):
            if i * 100 + j not in blocks:
                tmp = [neigh for neigh in map_datas[0][i][j].keys() if neigh not in blocks]
            else:
                tmp = []
            maps.append(tmp)
    return maps




# blocks = [201, 300, 302, 303, 304, 406, 405, 403]
# maps = map(9, 9, blocks)
# pos1 = (1, 1)
# pos2 = (8, 6)
# start_grid = Grid(pos1[0], pos1[1], maps)
# end_grid = Grid(pos2[0], pos2[1], maps)
# result_grid = a_star_search(start_grid, end_grid, maps)

# path = []
# while result_grid is not None:
#     path.append(Grid(result_grid.x, result_grid.y, maps))
#     result_grid = result_grid.parent
# print("path:")
# path.reverse()
# path = [[coord.x, coord.y] for coord in path]
# print(path)

# map(5, 5, [202])
# # 绘制蜂巢形状
# plt.figure(figsize=(8, 6))

# plot_hive(9, 9, 1, path, blocks)  # 5行7列的蜂巢，每个六边形大小为1
# plt.axis('equal')  # 设置坐标轴比例相等，使图形更符合比例
# plt.axis('off')    # 关闭坐标轴显示
# plt.show()
# 2526 2425
# [2932, 3032, 3132, 3541, 3542, 3643, 3932, 4032, 4131]
def Astar(pos1, pos2, roadblocks):
    maps = map(roadblocks)
    start_grid = Grid(pos1 // 100, pos1 % 100, maps)
    end_grid = Grid(pos2 // 100, pos2 % 100, maps)
    result_grid = a_star_search(start_grid, end_grid, maps)
    path = []
    while result_grid is not None:
        path.append(Grid(result_grid.x, result_grid.y, maps))
        result_grid = result_grid.parent
 
    path.reverse()
    path = [coord.x * 100 + coord.y for coord in path]
 
    return path

# path = Astar(2526, 2328, [2932, 3032, 3132, 3541, 3542, 3643, 3932, 4032, 4131])
# blocks = [2932, 3032, 3132, 3541, 3542, 3643, 3932, 4032, 4131]
# # 绘制蜂巢形状
# plt.figure(figsize=(8, 6))

# plot_hive(92, 77 , 1, path, blocks)  # 5行7列的蜂巢，每个六边形大小为1
# plt.axis('equal')  # 设置坐标轴比例相等，使图形更符合比例
# plt.axis('off')    # 关闭坐标轴显示
# plt.show()
# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import pickle
from Astar import Astar
from tqdm import tqdm


operators2idx = {
        0: 0,
        1: 1,
        2: 2,
        100: 3, 
        200: 4,  
        400: 5, 
        700: 6, 
        701: 7, 
        10000: 8, 
        10001: 9, 
        10100: 10, 
        10101: 11, 
        10200: 12, 
        10201: 13,
    }
sub_type2idx = {
        0: 0,  # 重型坦克 
        1: 1,  # 重型战车 中型战车
        2: 2,  # 步兵小队
        4: 3,  # 无人战车
        7: 4,  # 巡飞弹
    }


def removeFile():
    files = os.listdir('./dataset')
    idx = 1
    for fileName in files:
        arrs = deleteSingleFile(fileName)
        with open('./data/'+str(idx)+'.json', 'w', encoding='utf-8') as f:
            json.dump(arrs, f, indent=4)
        idx += 1
        
def deleteSingleFile(fileName: str):
    with open('./dataset/' + fileName, 'r', encoding='gb2312') as fp:
        datas = json.load(fp)
        arrs = []
        init = datas[0]['operators']
        arrs.append(datas[0])
        pre = {data['obj_id']: data['cur_hex'] for data in init}
        for i in range(1, len(datas)):
            cur = {data['obj_id']: data['cur_hex'] for data in datas[i]['operators']}
            if isSame(pre, cur):
                continue
            else:
                arrs.append(datas[i])
                pre = cur
        return arrs
            
def isSame(pre: dict, cur: dict):
    """
    判断是否有重复的态势图
    """
    if len(pre) != len(cur):
        return False
    
    for id in pre.keys():
        if id not in cur.keys() or pre[id] != cur[id]:
            return False
    return True
        
def singleFileNodeFeatures(fileName):
    """
    单个文件数据的节点特征，维度为[态势图数量，算子数量，算子类型]
    算子类型包含6种:重型坦克，重型战车/中型战车，步兵小队，无人战车，巡飞弹，被摧毁
    """
    with open('./data/' + fileName, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        init = datas[0]['operators']
        pre = np.zeros((len(operators2idx), 6))
        pre_idxs = [(operators2idx[data['obj_id']], sub_type2idx[data['sub_type']]) for data in init]
        for opeId, typeId in pre_idxs:
            pre[opeId][typeId] = 1
        arr = [pre]
        for i in range(1, len(datas)):
            now = np.zeros((len(operators2idx), 6))
            # 去掉炮兵信息
            new_idxs = [(operators2idx[data['obj_id']], sub_type2idx[data['sub_type']]) for data in datas[i]['operators'] if data['obj_id']!=300 and data['obj_id']!=301]
            # 消失算子最后一个特征标记为1
            
            for i, (opeId, typeId) in enumerate(pre_idxs):
                if pre_idxs[i][0] not in list(np.array(new_idxs)[:, 0]):
                    now[opeId][5] = 1
                    
            for opeId, typeId in new_idxs:
                    now[opeId][typeId] = 1

            pre_idxs = new_idxs
            arr.append(now)
        
        arr = np.concatenate([arr], axis=0)
        return arr   

def allNodeFeatures():
    fileNames = os.listdir('./data')
    idx = 1 
    for file in fileNames:
        arr = singleFileNodeFeatures(file)
        np.save('./node_features/' + str(idx), arr)
        idx += 1
        
def edgefeatures1():
    fileNames = os.listdir('./data')
    idx = 1
    for file in fileNames:
        arr = singleFileEdgeFeatures1(file)
        np.save('./edge_features1/' + str(idx), arr) 
        idx += 1

def singleFileEdgeFeatures1(fileName):
    with open('./data/' + fileName, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        edge_feature1 = []
        
        for i in range(len(datas)):
            operators = []
            locations = []
            for j in range(len(datas[i]['operators'])):
                operators.append(datas[i]['operators'][j]['obj_id'])
                locations.append(datas[i]['operators'][j]['cur_hex'])
            
            edge1 = np.zeros((14, 14))
            for j, operator1 in enumerate(operators):
                if operator1 not in operators2idx.keys():
                    continue
                for k, operator2 in enumerate(operators):
                    if operator2 not in operators2idx.keys():
                        continue
                    idx1 = operators2idx[operator1]
                    idx2 = operators2idx[operator2]
                    pos1 = locations[j]
                    pos2 = locations[k]
                    edge1[idx1][idx2] = manhattan(pos1, pos2)
            edge_feature1.append(edge1)
        return np.concatenate([edge_feature1], axis=0)    
    
def manhattan(pos1, pos2):
    pos1, pos2 = int(pos1), int(pos2)
    pos1_x, pos1_y = pos1 // 100, pos1 % 100
    pos2_x, pos2_y = pos2 // 100, pos2 % 100
    return abs(pos1_x - pos2_x) + abs(pos2_y - pos1_y)
        
def see(a_position, b_position):
    pos1, pos2 = int(a_position), int(b_position)
    pos1_x, pos1_y = pos1 // 100, pos1 % 100
    pos2_x, pos2_y = pos2 // 100, pos2 % 100
    seedata = np.load('./maps/map_53/53see.npz')['data'][0]
    return int(seedata[pos1_x, pos1_y, pos2_x, pos2_y])

def singleFileEdgeFeature2(fileName):    
    with open('./data/' + fileName, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        edge_feature2 = []
        
        for i in range(len(datas)):
            operators = []
            locations = []
            for j in range(len(datas[i]['operators'])):
                operators.append(datas[i]['operators'][j]['obj_id'])
                locations.append(datas[i]['operators'][j]['cur_hex'])
            
            edge1 = np.zeros((14, 14))
            for j, operator1 in enumerate(operators):
                if operator1 not in operators2idx.keys():
                    continue
                for k, operator2 in enumerate(operators):
                    if operator2 not in operators2idx.keys():
                        continue
                    idx1 = operators2idx[operator1]
                    idx2 = operators2idx[operator2]
                    pos1 = locations[j]
                    pos2 = locations[k]
                    edge1[idx1][idx2] = see(pos1, pos2)
            edge_feature2.append(edge1)
        return np.concatenate([edge_feature2], axis=0)

def edgefeatures2():
    fileNames = os.listdir('./data')
    idx = 1
    for file in fileNames:
        arr = singleFileEdgeFeature2(file)
        np.save('./edge_features2/' + str(idx), arr) 
        idx += 1
        
def calcCost(pos1, pos2, roadblock):
    graph = json.load(open('./maps/map_53/basic.json', 'r', encoding='gb2312'))['map_data']
    graphs = []
    for i in range(92):
        for j in range(77):
            graphs.append([neighbor for neighbor in graph[i][j]['neighbors'] if neighbor >= 0 and neighbor not in roadblock])
    
    path = Astar(pos1, pos2, roadblock)
    cost = 0
    costs = pickle.load(open('./maps/map_53/cost.pickle', 'rb'))
    for i in range(len(path)-1):
        x, y = path[i] // 100, path[i] % 100
        cost += costs[0][x][y][path[i+1]]
    return cost

def edgefeature3():
    fileNames = os.listdir('./data')
    idx = 40
    for file in fileNames[40:]:
        arr = singleFileEdgeFeature3(file)
        np.save('./edge_features3/' + str(idx), arr) 
        idx += 1
        
def singleFileEdgeFeature3(file):
    with open('./data/' + file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        edge_feature3 = []
        roadblock = datas[0]['landmarks']['roadblocks']
        print(len(datas))
        for i in tqdm(range(len(datas))):
            operators = []
            locations = []
            for j in range(len(datas[i]['operators'])):
                operators.append(datas[i]['operators'][j]['obj_id'])
                locations.append(datas[i]['operators'][j]['cur_hex'])
            
           
            edge1 = np.zeros((14, 14))
            for j, operator1 in enumerate(operators):
                print('j:', j)
                if operator1 not in operators2idx.keys():
                    continue
                for k, operator2 in enumerate(operators):
                    if operator2 not in operators2idx.keys():
                        continue
                    idx1 = operators2idx[operator1]
                    idx2 = operators2idx[operator2]
                    pos1 = locations[j]
                    pos2 = locations[k]
                    edge1[idx1][idx2] = calcCost(pos1, pos2, roadblock)
            edge_feature3.append(edge1)
        return np.concatenate([edge_feature3], axis=0)


def graphdata():
    graph = json.load(open('./maps/map_53/basic.json', 'r', encoding='gb2312'))['map_data']
    graphs = []
    for i in range(92):
        for j in range(77):
            graphs.append([neighbor for neighbor in graph[i][j]['neighbors']])
    return graphs

if __name__ == '__main__':
    # removeFile()  # 删除冗余文件
    # allNodeFeatures()  # 节点特征矩阵
    # edgefeatures1()
    # edgefeatures2()
    # data = np.load('./maps/map_21/21see.npz')
    # print(data['data'].shape)
    # data1 = json.load(open('./maps/map_29/basic.json', 'r', encoding='gb2312'))
    # print(data1.keys())
    # print(data1['map_data'][29][32])
    # print(len(data1['map_data']))
    # print(len(data1['map_data'][0]))
    # print(data1['map_id'])
    # with open('./data/' + '1.json', 'r', encoding='utf-8') as f:
    #     datas = json.load(f)
    #     edge_feature3 = []
    #     roadblock = datas[0]['landmarks']['roadblocks']
    #     print(roadblock)
    # import pickle
    # data2 = pickle.load(open('./maps/map_53/cost.pickle', 'rb'))
    # print(len(data2))
    # print(len(data2[0]))
    # print(len(data2[0][0]))
    # print((data2[0][29][33]))
    # print((data2[2][25][26]))
    #print(data1['map_data'])
    # dict_keys(['scenario_id', 'operators', 'time', 'cities', 'landmarks', 'blueprints', 'annual_version'])
    # data4 = json.load(open('./scenarios/2110431453.json', 'r'))
    # print(data4['annual_version'])
    # print('--------------------------------------------------')
    # graph = graphdata()
    # print(len(graph))
    edgefeature3()
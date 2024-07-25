import argparse
import random
import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json

from model import GGNN
from data import Data
from create_data import calcCost
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--gru_dim', type=int, default=30, help='GRU output state size')
    parser.add_argument('--state_dim', type=int, default=50, help='GGNN hidden state size')
    parser.add_argument('--n_steps', type=int, default=3)
    parser.add_argument('--niter', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    return args

def set_seed(args):
    random.seed(2023)
    torch.manual_seed(2023)
    if args.cuda:
        torch.cuda.manual_seed_all(2023)

def main(args):
    node_feats_files = '/home/wwh/test/node_features'
    edge1_feats_files = '/home/wwh/test/edge_features1'
    edge2_feats_files = '/home/wwh/test/edge_features2'
    edge3_feats_files = '/home/wwh/test/edge_features3'
    
    dataset = Data([node_feats_files, edge1_feats_files, edge2_feats_files, edge3_feats_files])
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=2023)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers)
   
    args.n_edge_types = 3
    args.n_nodes = dataset.n_node
    args.annotation_dim = dataset.node_features_dim
    print(args)

    net = GGNN(opt=args)

    criterion = nn.MSELoss()

    if args.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.01)
    net.load_state_dict(torch.load('./checkpoints/45.pth'))
    for epoch in range(0, args.niter):
        train(epoch, train_dataloader, net, criterion, optimizer, args)    
        test(test_dataloader, net, criterion, args)
        if epoch > 10 and epoch % 5 == 0:
            torch.save(net.state_dict(), f'./checkpoints/{epoch}.pth')

def train(epoch, dataloader, net, criterion, optimizer, args):
    net.train()
    for i, (node_feats, edge_feats, node1_feats, edge1_feats) in enumerate(dataloader, 1):
        
        net.zero_grad()
        padding = torch.zeros(len(node_feats), args.n_nodes, args.state_dim-args.annotation_dim)
        input = torch.cat((node_feats, padding), 2)

        if args.cuda:
            input = input.cuda()
            edge_feats = edge_feats.cuda()
            node1_feats = node1_feats.cuda()
            edge1_feats = edge1_feats.cuda()
        
        new_node_feats, new_edge_feats, mu, logvar = net(input, edge_feats)
        # print(new_node_feats)
        kld = 0.5 * torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1. - logvar)
        #print(edge_feats)
        # print(new_edge_feats)
        print(new_node_feats[0])
        print('old', node1_feats[0])
        loss1 = criterion(new_edge_feats, edge1_feats)
        loss2 = criterion(new_node_feats, node1_feats)
        
        # node_feats = torch.argmax(node_feats[0], dim = 1)
        # new_node_feats = torch.argmax(new_node_feats[0], dim =1)

        loss = loss1 + loss2 + kld

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            # print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, args.niter, i, len(dataloader), loss.item()))
            print('[%d/%d][%d/%d] Loss: %.4f, Loss1: %.4f, Loss2: %.4f, kld: %.4f' % (epoch, args.niter, i, len(dataloader), loss.item(),loss1.item(), loss2.item(), kld.item()))

def test(dataloader, net, criterion, args):
    test_loss = 0
    net.eval()
    for i, (node_feats, edge_feats, node1_feats, edge1_feats) in enumerate(dataloader):
        padding = torch.zeros(len(node_feats), args.n_nodes, args.state_dim-args.annotation_dim)
        input = torch.cat((node_feats, padding), 2)

        if args.cuda:
            input = input.cuda()
            edge_feats = edge_feats.cuda()
            node1_feats = node1_feats.cuda()
            edge1_feats = edge1_feats.cuda()
        
        new_node_feats, new_edge_feats, mu, logvar = net(input, edge_feats)

        kld = 0.5 * torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1. - logvar)
        loss1 = criterion(new_edge_feats, edge1_feats)
        loss2 = criterion(new_node_feats, node1_feats)
        loss = loss1 + loss2 + kld

        test_loss += loss.item()
    test_loss /= len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))

def predict(edge_features, red_positions, blue_idx, file):
    # red_positions : 7维向量
    # edge_features:[14, 14, 3], node_features:[14, 6]
    # blue_position: 标量一个维度, 表示蓝方算子的索引
    # file: 该态势图在的数据文件，xx.npy
    red_objs = [2, 8, 9, 10, 11, 12, 13]
    edge_features = torch.squeeze(edge_features).reshape(14, 14, 3)
    # print(edge_features[:, :, 0])
    predict_distances = []
    predict_sees = []
    predict_costs = []
    for red_obj in red_objs:
        predict_distances.append(edge_features[blue_idx][red_obj][0])
        predict_sees.append(edge_features[blue_idx][red_obj][1])
        predict_costs.append(edge_features[blue_idx][red_obj][2])


    red2truth_distances = []
    red2truth_sees = []
    red2truth_costs = []

    seedata = np.load('./maps/map_53/53see.npz')['data'][0]
    with open('./data/' + file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        roadblock = datas[0]['landmarks']['roadblocks']
    for k in range(len(red_positions)):
        distances = []
        sees = []
        costs = []
        red_x = red_positions[k] // 100
        red_y = red_positions[k] % 100
        for i in range(21, 59):
            for j in range(18, 54):
                distances.append(1 - abs(predict_distances[k] - (abs(i - red_x) + abs(j - red_y))) / predict_distances[k])

                if seedata[red_x, red_y, i, j]: 
                    sees.append(1.)
                else:
                    sees.append(0.5)
                #costs.append(2 - predict_costs[k] / calcCost(i * 100 + j, red_positions[k], roadblock))

        distances = torch.tensor(distances)
        sees = torch.tensor(sees)
        costs = torch.tensor(costs)

        red2truth_distances.append((distances - min(distances))/(max(distances) - min(distances)))
        red2truth_sees.append(sees)
        #red2truth_costs.append((costs - min(costs)) / (max(costs) - min(costs)))

    red2truth_distances = torch.tensor([item.numpy() for item in red2truth_distances])
    red2truth_sees = torch.tensor([item.numpy() for item in red2truth_sees])
    #red2truth_costs = torch.tensor([item.numpy() for item in red2truth_costs])


    #predict = torch.sum(torch.mul(torch.mul(red2truth_sees,red2truth_costs), red2truth_distances), dim=0)
    # np.save( '/home/wwh/test/distance.npy',red2truth_distances.numpy())
    predict = torch.sum(torch.mul(red2truth_sees, red2truth_distances), dim=0)
    # predict = torch.reshape(predict, [38, 36])
    # print(predict)

    # print(predict)
    topk_values, topk_indices = torch.topk(predict, 10)
    positions = []
    for i in range(21, 59):
            for j in range(18, 54):
                positions.append(i * 100 + j)
    positions = torch.tensor(positions)
    return positions[topk_indices]

def main_predict(args):
    node_feats_files = '/home/wwh/test/node_features'
    edge1_feats_files = '/home/wwh/test/edge_features1'
    edge2_feats_files = '/home/wwh/test/edge_features2'
    edge3_feats_files = '/home/wwh/test/edge_features3'

    red_positions = []
    with open('./data/1.json') as f:
        datas = json.load(f)
        red_obj = [2, 10000, 10001, 10100, 10101, 10200, 10201]
        red_positions = []
        for data in datas[0]['operators']:
            if data['obj_id'] in red_obj:
                red_positions.append(data['cur_hex'])
    
    dataset = Data([node_feats_files, edge1_feats_files, edge2_feats_files, edge3_feats_files])
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=2023)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
   
    args.n_edge_types = 3
    args.n_nodes = dataset.n_node
    args.annotation_dim = dataset.node_features_dim

    net = GGNN(opt=args)
    net.load_state_dict(torch.load('./checkpoints/45.pth'))
    for i, (node_feats, edge_feats, node1_feats, edge1_feats) in enumerate(test_dataloader):
        padding = torch.zeros(len(node_feats), args.n_nodes, args.state_dim-args.annotation_dim)
        input = torch.cat((node_feats, padding), 2)

        if args.cuda:
            input = input.cuda()
            edge_feats = edge_feats.cuda()
            node1_feats = node1_feats.cuda()
            edge1_feats = edge1_feats.cuda()
        print(edge1_feats[0, :, :14])
        print('-----------------')
      
        new_node_feats, new_edge_feats, mu, logvar = net(input, edge_feats)
        print(new_edge_feats[0, :,:14])
        positions = predict(new_edge_feats, red_positions, 1, '1.json')
        print('----------------------')
        break

if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    # main(args)
    main_predict(args)
    


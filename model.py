import torch.nn as nn
import torch


class AttrProxy():
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class GGNN(nn.Module):
    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim, 'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_nodes
        self.n_steps = opt.n_steps
        print('edge_type:', self.n_edge_types)

        for i in range(self.n_edge_types):
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, 'in_')
        self.out_fcs = AttrProxy(self, 'out_')

        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)
        self.mu = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim)
        )
        self.var = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim)
        )
        self.dec1 = nn.Sequential(
            nn.Linear(self.state_dim, self.annotation_dim),
            # nn.Softmax(dim=2)
        )
        self.dec2 = nn.Sequential(
            nn.Linear(self.state_dim, self.n_node * self.n_edge_types),
            # nn.Sigmoid()
        )
        # self._initialization()



    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, prop_state, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            # [b, 2, 4, 8] = [b, edge_types, node, state_dim]
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node * self.n_edge_types, self.state_dim)
            # [b, 8, 8]
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node * self.n_edge_types, self.state_dim)
            # [b, 4, 8], [b, 8, 8], [b, 4, 8], [b, 4, 16]
            # [b, 28, 50], [b, 28, 50], [32, 14, 50], [32, 14, 28]
            prop_state = self.propogator(in_states, out_states, prop_state, A)

        
        join_state = prop_state
        mu = self.mu(join_state)
        var = self.var(join_state)
        # [16, 4, 8] 
        z = self.reparameterize(mu, var)

        node_feats = self.dec1(z)
        edge_feats = self.dec2(z)
        # output = output.sum(2)

        return node_feats, edge_feats, mu, var
    
class Propogator(nn.Module):
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()
        self.n_node = n_node
        self.n_edge_types = n_edge_types
        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Sigmoid()
        )
    
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Sigmoid()
        )

        self.tansform = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node * self.n_edge_types]
        # A_out = A[:, :, self.n_node * self.n_edge_types:]

        # [16, 4, 8] [16, 8, 8]
        
        a_in = torch.bmm(A_in, state_in)
        # a_out = torch.bmm(A_out, state_out)
        # [16, 4, 8], [16, 4, 8]
        a = torch.cat((a_in, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output



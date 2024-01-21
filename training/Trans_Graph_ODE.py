import torch_geometric
from torch_geometric import nn
import torch
import math
import numpy as np
from torchdiffeq._impl import odeint
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence,Independent

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.size()) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res
def reparameterize(mu, var):
    """
    Samples z from a multivariate Gaussian with diagonal covariance matrix using the
    reparameterization trick.
    """
    d = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda())
    r = d.sample(mu.size()).squeeze(-1)
    return r * var.float() + mu.float()

def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=std)
            torch.nn.init.constant_(m.bias, val=0)

class Relation_Hyper(torch.nn.Module):
    def __init__(self, layer_num, emb_size=100):
        super(Relation_Hyper, self).__init__()
        self.emb_size=emb_size
        self.layer_num=layer_num
    def forward(self,adjacency, embedding):
        item_embeddings=embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0.detach().cpu()]
        for i in range(self.layer_num):
            item_embeddings = torch.sparse.mm(adjacency.cuda(), item_embeddings)
            final.append(item_embeddings.cpu().detach().numpy())
        item_embeddings = np.sum(final,0)
        return item_embeddings

class Line_Convolution(torch.nn.Module):
    def __init__(self,layers,emb_size=100):
        super(Line_Convolution, self).__init__()
        self.emb_size=emb_size
        self.layers=layers
    def forward(self, item_embedding, D, A):
        session = [item_embedding.detach().cpu()]
        DA = torch.mm(D, A).float().detach().cpu()
        session_emb_lgcn=item_embedding.detach().cpu()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn.cpu().detach().numpy())
        session_emb_lgcn = np.sum(session, 0)
        return session_emb_lgcn

class Stock_Hyper(torch.nn.Module):
    def __init__(self,adjacency,emb_size, n_node,l2,lr,layers, beta):
        super(Stock_Hyper, self).__init__()
        self.emb_size = emb_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.adjacency = adjacency
        self.embedding = torch.nn.Embedding(self.n_node, self.emb_size)
        self.embedding1 = torch.nn.Embedding(1142, self.emb_size)
        self.HyperGraph = Relation_Hyper(self.layers)
        self.LineGraph = Line_Convolution(self.layers)
        self.w_1 = torch.nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = torch.nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.init_parameters()
        self.line1 = torch.nn.Linear(self.emb_size, 64)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, D, A):
        item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight).cuda()
        session_emb_lg = self.LineGraph(self.embedding1.weight, D, A,).cuda()
        out=torch.transpose(session_emb_lg,1,0).cuda()
        out1=torch.mm(item_embeddings_hg,out).cuda()
        #out1=self.line1(item_embeddings_hg).cuda()
        return out1

class atten_init(torch.nn.Module):
    def __init__(self, dim):
        super(atten_init, self).__init__()
        self.norm1 = torch.nn.InstanceNorm1d(dim)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.conv1 = torch.nn.Linear(dim, dim)
        self.norm2 = torch.nn.InstanceNorm1d(dim)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.norm3 = torch.nn.InstanceNorm1d(dim)

    def forward(self, x):
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        return out

class ODEfunc(torch.nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv1 = torch.nn.Linear(dim//2, dim)
        self.conv2 = torch.nn.Linear(dim, dim)
        self.relu_at = torch.nn.ReLU(inplace=False)
        self.conv1_at = torch.nn.Linear(dim//2, dim)
        self.conv2_at = torch.nn.Linear(dim, dim)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.nfe = 0
    def forward(self, t, x):
        size = x.shape
        size = int(size[1]/2)
        h_0 = x[:,:size]
        a_0 = x[:,size:]
        a_s = self.sigmoid1(a_0)
        h_ = torch.mul(h_0, a_s)
        self.nfe += 1
        out1 = self.relu(h_)
        out1 = self.conv1(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)
        out2 = self.relu_at(h_)
        out2 = self.conv1_at(out2)
        out2 = self.relu_at(out2)
        out2 = self.conv2_at(out2)
        out = torch.cat([out1, out2])
        return out1

class ODEBlock(torch.nn.Module):
    def __init__(self, odefunc, atten_init):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.atten_init = atten_init

        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        a_0 = self.atten_init(x)
        x_0 = torch.cat([x, a_0],-1)
        out = odeint(self.odefunc, x_0, self.integration_time,
                    rtol=1e-6, atol=1e-6,method='euler')
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class VAETransformerEncoder(torch.nn.Module):
  def __init__(self, n_layer, n_head, d_model=9, d_ff=128, d_vae_latent=64, dropout=0.1, activation='relu'):
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation

    self.tr_encoder_layer = torch.nn.TransformerEncoderLayer(
      d_model, n_head, d_ff, dropout, activation
    )
    self.tr_encoder = torch.nn.TransformerEncoder(
      self.tr_encoder_layer, n_layer
    )

    self.fc_mu = torch.nn.Linear(d_model, d_vae_latent)
    self.fc_logvar = torch.nn.Linear(d_model, d_vae_latent)

  def forward(self, x, padding_mask=None):
    out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
    hidden_out = out
    mu, logvar = self.fc_mu(hidden_out), self.fc_logvar(hidden_out)
    return hidden_out, mu, logvar


class RESDecoder(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64):
        super(RESDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.gru = torch.nn.GRU(in_dim,hid_dim,batch_first=True,dropout=0.1,num_layers=2)

    def forward(self, input):

        output, hidden = self.gru(input)
        return output,hidden
class Decoder(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.line= torch.nn.Linear(in_dim,hid_dim)

    def forward(self, input):
        output = self.line(input)
        return output


class Stock_ODE(torch.nn.Module):
    def __init__(self,Stock_num,in_dim,emb_size,days,n_layer=1, n_head=1, dropout_rate=0.1):
        super(Stock_ODE, self).__init__()
        self.emb_size=emb_size
        self.in_dim = in_dim
        self.n_layer=n_layer
        self.n_head=n_head
        self.days = days
        self.decoder = Decoder(in_dim,emb_size)
        #self.res_decoder=RESDecoder(in_dim,emb_size)
        self.res_decoder = RESDecoder(in_dim, emb_size)
        self.encoder=VAETransformerEncoder(self.n_layer,self.n_head)
        self.dropout_rate = dropout_rate
        self.atte=atten_init(emb_size)
        self.ode = ODEBlock(ODEfunc(emb_size*2),self.atte)
        self.line=torch.nn.Linear(9,64)
        self.line1 = torch.nn.Linear(128, 64)
        self.mean_dense = torch.nn.Linear(192,64)
        self.W = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(4,1026, 64), std=0.1).cuda())
        self.output_in_dim=64
        self.output_out_dim=64
        self.std_dense = torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.Softplus(), torch.nn.Hardtanh(min_val=0.01, max_val=7.))
        self.encoder1=torch.nn.GRU(9,64,batch_first=True,dropout=0.1,num_layers=2)


        self.output =torch.nn.Sequential(
            torch.nn.Linear(64, self.output_in_dim // 2),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(self.output_in_dim // 2, self.output_in_dim // 4),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(self.output_in_dim // 4, self.output_out_dim),
        )


    def forward(self,inputs,moving):
        x=inputs
        x=torch.transpose(x,1,0)
        x=x*moving
        x=x*self.W
        hidden_out, mu, logvar = self.encoder(x)
        #encoder_outputs = torch.cat([mu, hidden_out, logvar], -1)
        encoder_outputs1 = self.line(hidden_out)
        #encoder_outputs1,hidden_out=self.encoder1(x)
        all_hiddens = []
        all_z = []
        prev_z_mean = torch.zeros((1026, 64)).cuda()
        prev_z_std = torch.zeros((1026, 64)).cuda()
        for index,data in enumerate(encoder_outputs1):
            hidden, hidden_std=data,data
            #hidden, hidden_std=self.encoder1(data)
            all_hiddens.append(hidden)
            z_mean =self.ode(prev_z_mean)
            #z_mean = self.line1(z_mean)
            z_std = hidden_std
            z_mean = self.mean_dense(torch.cat([hidden,z_mean], dim=1))+prev_z_mean
            #print(z_mean.size())
            z_std = self.std_dense(torch.cat([z_std, prev_z_std], dim=1))
            z_std = z_std.abs()
            prev_z_mean, prev_z_std = z_mean, z_std
            all_z.append(z_mean)
        z = reparameterize(z_mean, z_std)
        z_distr = Normal(z_mean, z_std)
        z_prior = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda())
        kl_z = kl_divergence(z_distr, z_prior)
        kl_z = torch.sum(kl_z)
        kl_all = kl_z / float(1026)
        all_hiddens = torch.stack(all_hiddens, 0)
        all_hiddens = all_hiddens.permute(1, 0, 2)
        all_z = torch.stack(all_z, 0)
        all_z = all_z.permute(1, 0, 2)
        z_p = self.decoder(z)
        input_res, c = self.res_decoder(all_z)
        loss_fn = torch.nn.MSELoss()
        res_loss = loss_fn(all_hiddens, input_res)
        pred = self.output(z_p)
        gaussian = Independent(Normal(loc=pred, scale=0.1), 1)
        return pred, gaussian, kl_all, res_loss

class Stock_Hyper_ODE(torch.nn.Module):
    def __init__(self,adj,Stock_num,n_node):
        super(Stock_Hyper_ODE, self).__init__()
        self.adj=adj
        self.Stock_num=Stock_num
        self.in_dim=64
        self.emb_size=64
        self.days=4
        self.l2=0.0001
        self.lr=0.0001
        self.layers=1
        self.beta=0.15
        self.z1=Stock_ODE(self.Stock_num,self.in_dim,self.emb_size,self.days)
        self.z2=Stock_Hyper(self.adj,self.emb_size, n_node,self.l2,self.lr,self.layers, self.beta)
        self.line1 = torch.nn.Linear(128, 1)
        #self.line1 = torch.nn.Linear(64, 1)
        self.line2=torch.nn.Linear(1142,64)
    def forward(self,inputs,moving,D,A):
        x=inputs
        pred, gaussian, kl_all, res_loss=self.z1(x,moving)
        out2=self.z2(D,A)
        out2=self.line2(out2)
        out3=torch.cat([pred,out2],-1)
        out=self.line1(out3)
        return out,gaussian,kl_all,res_loss









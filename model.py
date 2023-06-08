import torch
from torch import nn
from torchmetrics.functional import mean_absolute_percentage_error

class Representation(nn.Module):
    # 自V生成C和Z的网络结构
    def __init__(self, input_dim, output_dim, n_layer):
        super().__init__()
        layers = [nn.Linear(input_dim, output_dim)]
        for i in range(n_layer-1):
            layers.append(nn.Linear(output_dim, output_dim))
        self.linear = nn.Sequential(*layers)
        self.norm = nn.BatchNorm1d(input_dim)
            
    def forward(self, input):
        output = self.norm(input)
        output = self.linear(output)
        return output

class RegressionY(torch.nn.Module):
    # Y对X和C的回归
    def __init__(self, c_dim, x_dim, emb_dim, output_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(x_dim)
        self.norm2 = nn.BatchNorm1d(c_dim)
        self.embedding = nn.Linear(x_dim, emb_dim)
        self.linear = nn.Linear(c_dim+emb_dim*2, output_dim)
        self.core = nn.Sigmoid()

    def forward(self, x, c):
        output_x = self.norm1(x)
        output_c = self.norm2(c)
        output = self.embedding(output_x)
        output = torch.cat((output, output**2, output_c), 1)
        output = self.linear(output)
        output = self.core(output)
        return output

class RegressionX(nn.Module):
    # X对Z和C的回归
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim*2, output_dim)
        self.core = nn.Sigmoid()

    def forward(self, input):
        output = self.norm(input)
        output = torch.cat((output, output**2), 1)
        output = self.linear(output)
        output = self.core(output)
        return output

class MINet(nn.Module):
    def __init__(self, X_dim, Y_dim, randomized=False):
        super().__init__()
        self.norm = nn.BatchNorm1d(X_dim)
        self.mu = nn.Sequential(
            nn.Linear(X_dim, X_dim//2),
            nn.ELU(),
            nn.Linear(X_dim//2, Y_dim)
        )
        self.logvar = nn.Sequential(
            nn.Linear(X_dim, X_dim//2),
            nn.ELU(),
            nn.Linear(X_dim//2, Y_dim),
            nn.Tanh()
        )
        self.randomized = randomized

    def get_mu_logvar(self, X):
        X = self.norm(X)
        mu = self.mu(X)
        logvar = self.logvar(X)
        return mu, logvar
    
    def loglikelihood(self, X, Y):
        mu, logvar = self.get_mu_logvar(X)
        loglikelihood = (-(mu - Y)**2 / logvar.exp() - logvar).sum(1).mean(0)
        return loglikelihood

    def forward(self, X, Y):
        mu, logvar = self.get_mu_logvar(X)
        positive = -(mu-Y)**2 / 2 / logvar.exp()
        if (self.randomized):
            random_index = torch.randperm(X.shape[0]).long()
            negative = -(mu - Y[random_index])**2 / 2 / logvar.exp()
        else:
            negative = -((Y.unsqueeze(0) - mu.unsqueeze(1)**2)).mean(dim=1) / 2 / logvar.exp()
        mi = -(positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return mi

class MIWeightedNet(nn.Module):
    def __init__(self, X_dim, Y_dim, sigma=0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(X_dim)
        self.mu = nn.Sequential(
            nn.Linear(X_dim, X_dim//2),
            nn.ELU(),
            nn.Linear(X_dim//2, Y_dim)
        )
        
        self.logvar = nn.Sequential(
            nn.Linear(X_dim, X_dim//2),
            nn.ELU(),
            nn.Linear(X_dim//2, Y_dim),
            nn.Tanh()
        )

        self.sigma = 0.5

    def get_mu_logvar(self, X):
        X = self.norm(X)
        mu = self.mu(X)
        logvar = self.logvar(X)
        return mu, logvar
    
    def loglikelihood(self, X, Y):
        mu, logvar = self.get_mu_logvar(X)
        loglikelihood = (-(mu - Y)**2 / logvar.exp() - logvar).sum(1).mean(0)
        return loglikelihood

    def forward(self, X, Y, inter_var):
        mu, logvar = self.get_mu_logvar(X)
        positive = -(mu-Y)**2 / 2 / logvar.exp()
        negative = -(((Y.unsqueeze(0) - mu.unsqueeze(1))**2)).mean(dim=1) / 2 / logvar.exp()
        w = torch.exp(((inter_var.unsqueeze(1) - inter_var.unsqueeze(0)).sum(-1))**2 / 2 / self.sigma**2)
        w = nn.functional.softmax(w, dim=1)
        mi = -(w * (positive.sum(dim = -1) - negative.sum(dim = -1).unsqueeze(1))).mean()
        return mi

class AutoIV(nn.Module):
    def __init__(self, x_dim, v_dim, y_dim, parameter):
        super().__init__()
        self.x_dim = x_dim
        self.v_dim = v_dim
        self.y_dim = y_dim

        self.z_dim = parameter['z_dim']
        self.c_dim = parameter['c_dim']
        self.emb_dim = parameter['emb_dim']
        self.lld_weight = parameter['lld_weight']
        self.mi_weight = parameter['mi_weight']
        self.n_layer = parameter['n_layer']
        self.alpha = parameter['alpha']
        self.eta = parameter['eta']

        self.z_rep = Representation(self.v_dim, self.z_dim, self.n_layer)
        self.c_rep = Representation(self.v_dim, self.c_dim, self.n_layer)
        self.regressionX = RegressionX(self.z_dim + self.c_dim, self.x_dim)
        self.regressionY = RegressionY(self.c_dim, self.x_dim, self.emb_dim, self.y_dim)
        self.naive_regression = nn.Linear(self.x_dim, self.y_dim)
        
        self.zx_minet = MINet(self.z_dim, self.x_dim)
        self.zy_minet = MIWeightedNet(self.z_dim, self.y_dim)
        self.cx_minet = MINet(self.c_dim, self.x_dim)
        self.cy_minet = MINet(self.c_dim, self.y_dim)
        self.zc_minet = MINet(self.z_dim, self.c_dim)
        
    def forward(self, v):
        c = self.c_rep(v)
        z = self.z_rep(v)
        return c, z

    def get_lld_loss(self, v, x, y):
        c, z = self.forward(v)
        lld_zx = self.zx_minet.loglikelihood(z, x)
        lld_zy = self.zy_minet.loglikelihood(z, y)
        lld_cx = self.cx_minet.loglikelihood(c, x)
        lld_cy = self.cy_minet.loglikelihood(c, y)
        lld_zc = self.zc_minet.loglikelihood(z, c)
        return -(lld_zx + lld_zy + lld_cx + lld_cy + lld_zc)

    def get_mi_loss(self, v, x, y):
        c, z = self.forward(v)
        mi_zx = self.zx_minet.forward(z, x)
        mi_zy = self.zy_minet.forward(z, y, inter_var=x)
        mi_cx = self.cx_minet.forward(c, x)
        mi_cy = self.cy_minet.forward(c, y)
        mi_zc = self.zc_minet.forward(z, c)

        return mi_zx + (-mi_zy) + self.alpha * (mi_cx + mi_cy) + self.alpha * (-mi_zc)

    def get_x_pred(self, v):
        c, z = self.forward(v)
        x_input = torch.cat((c,z), 1)
        x_pred = self.regressionX(x_input)
        return x_pred

    def get_y_pred(self, v):
        c, z = self.forward(v)
        x_pred = self.get_x_pred(v)
        y_pred = self.regressionY(x_pred, c)
        return y_pred

    def get_x_loss(self, v, x):
        x_pred = self.get_x_pred(v).squeeze()
        return ((x_pred-x)**2).mean()
        #return mean_absolute_percentage_error(x_pred, x)

    def get_y_loss(self, v, y):
        y_pred = self.get_y_pred(v).squeeze()
        #y_pred = self.naive_regression(v).squeeze()
        return ((y_pred-y)**2).mean()
        #return mean_absolute_percentage_error(y_pred, y)

    def get_weight(self, model):
        weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weights.append(weight)
        return weights

    def get_reg_penalty(self, model, l=0.5, p=2): 
        weights = self.get_weight(model)
        penalty = 0
        for weight in weights:
            penalty += torch.norm(weight, p=p)
        return l*penalty

    def get_mape(self, v, y):
        y_pred = self.get_y_pred(v).squeeze()
        return mean_absolute_percentage_error(y_pred, y)






import torch
from torch.autograd import Variable
import numpy as np


class Var:
    '''
        Variables in factor graphs
    '''
    def __init__(self, col, label, type_size, batch_size=1, device='cpu'):
        self.col = col # index of col in the graph
        self.label = label
        self.belief = Variable(torch.zeros(batch_size, type_size), requires_grad=True).to(device)


    def set_belief(self, belief):
        self.belief = belief

    def __str__(self):
        return "col: {}, label: {}".format(self.col, self.label)

class Factor:
    '''
        Factors in factor graphs.
        Supports unary factor and pariwise factors
    '''
    def __init__(self, kind, idx, var1, var2, type_size, batch_size=1, device='cpu'):
        self.kind = kind # unary or pairwise factor
        self.var1 = var1
        self.var2 = var2
        self.idx = idx

        if self.kind == 'unary':
            self.belief = None
        elif self.kind == 'pairwise':
            self.belief = Variable(torch.zeros((batch_size, type_size, type_size), requires_grad=True)).to(device)
        else:
            assert False, "Invalid kind for factors!"

    def set_belief(self, belief):
        self.belief = belief

    def get_other_var(self, var):
        if self.var1==var:
            return self.var2
        else:
            return self.var1 

class FactorGraph:
    '''
        Structure of factor graph
        Supports unary factor and pariwise factors
    '''
    def __init__(self, type_size, mask, batch_size, device):
        self.vars = []
        self.factors = []
        self.type_size = type_size
        self.batch_size = batch_size
        self.device = device
        self.var2factoridx = {} # mapping from var to factor idx
        self.col2varidx = {} # mapping from col to var idx

        assert mask.shape[0] == batch_size , "Valid mask shape must be (batch_size, MAX_COL_COUNT)"

        # Transpose the matrix to (max_col, batch_size)
        # so that batch for a col can be retrieved through mask[i]
        self.var_mask_T = mask.transpose(0,1).to(self.device)
        self.factor_mask_T = None

    def var_iter(self):
        for v in self.vars:
            yield v

    def factor_iter(self):
        for f in self.factors:
            yield f


    def add_variable(self, col, label):
        self.vars.append(Var(col, label, self.type_size, self.batch_size, self.device))
        var_idx = len(self.vars) - 1
        self.col2varidx[col] = var_idx

    def add_factor(self, kind, var1, var2):
        factor_idx = len(self.factors) 
        self.factors.append(Factor(kind, factor_idx, var1, var2, self.type_size, self.batch_size, self.device))
        # save mapping from vars to factor
        self.var2factoridx[var1] = self.var2factoridx.get(var1, []) + [factor_idx]
        self.var2factoridx[var2] = self.var2factoridx.get(var2, []) + [factor_idx]


    def update_factor_mask(self):
        factor_mask_T = []
        for f in self.factors: 
            if f.kind == 'unary':
                # get variable mask as factor mask
                factor_mask_T.append(self.var_mask_T[f.var1.col].unsqueeze(0))
            elif f.kind == 'pairwise':
                # factor will be masked if either of the connected var is masked
                factor_mask_T.append((self.var_mask_T[f.var1.col] * self.var_mask_T[f.var2.col]).unsqueeze(0))

        self.factor_mask_T = torch.cat(factor_mask_T).to(self.device)

    def get_var_by_factor(self, factor):
        return factor.var1, factor.var2

    def get_factor_by_var(self, var):
        # return factors that are connected to a single var
        return [self.factors[idx] for idx in self.var2factoridx[var]]

    def get_factor_by_vars(self, var1, var2):
        # return the factor connectin two vars
        # unary factor if var2 is UNARY_VAR
        assert var1!=var2, 'Vars of a factor can\'t be same'
        factor_idx = list(set(self.var2factoridx[var1]).intersection(self.var2factoridx[var2]))[0]
        return self.factors[factor_idx]

    def get_var_by_col(self, col):
        return self.vars[self.col2varidx[col]]

    def get_var_mask(self, transpose=False):
        assert self.var_mask_T is not None, "Var mask not set"
        return self.var_mask_T if transpose else self.var_mask_T.transpose(0,1)

    def get_factor_mask(self, transpose=False):
        if self.factor_mask_T is None:
            self.update_factor_mask()

        return self.factor_mask_T if transpose else self.factor_mask_T.transpose(0,1)

class Meassage:
    def __init__(self, frm, to, value):
        self.frm = frm
        self.to = to
        self.value = value

    def set_value(self, value):
        self.value = value 

class BatchMessages:
    '''
        All the messages in a batch for factor graph
    '''
    def __init__(self, graph, batch_size):
        self.messages = {}
        self.batch_size = batch_size

        for var in graph.vars:
            self.messages[var] = {}

        for factor in graph.factors:
            self.messages[factor] = {}

        # init messages with uniform values
        
        # factor -> variable
        for var in graph.vars:
            for factor in graph.get_factor_by_var(var):
                self.messages[factor][var] = {}
                self.add_msg(factor, var, np.full((batch_size, graph.type_size), np.log(1./graph.type_size)) )
        # variable -> factor
        for factor in graph.factors:
            if factor.kind != 'unary': # no msg from var to unary factors
                for var in graph.get_var_by_factor(factor):
                    self.messages[var][factor] = {}
                    self.add_msg(var, factor, np.full((batch_size, graph.type_size), np.log(1./graph.type_size)) )


    def __iter__(self):
        for frm in self.messages.keys():
            for msg in self.messages[frm].values():
                yield msg

    def get_msg(self, frm, to):
        return self.messages[frm][to]

    def add_msg(self, frm, to, value):
        self.messages[frm][to] = Meassage(frm, to, value)

    def update_msg(self, frm, to, value):
        msg = self.get_msg(frm, to)
        msg.set_value(value)




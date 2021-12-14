from torch import nn
import importlib
from torch import optim
from torch.nn import MarginRankingLoss

"""
Get loss function from str parameter
"""
def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """

    if args == "NNL":
       return nn.NLLLoss()
    if args == "CrossEntropy":
      return nn.CrossEntropyLoss(reduction='mean')
    else:
       print("Not implemented yet")


"""
Get model from str parameter
"""
def get_net(args):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(args)
    num_params = sum([param.nelement() for param in net.parameters()])
    print('Model params = {:2.1f}M'.format(num_params / 1000000))
    net = net.cuda()
    return net

def get_model(args):
    """
    Fetch Network Function Pointer
    """
    module='graphs.models.'+args.arch
    mod = importlib.import_module(module)
    net = getattr(mod, (args.arch.title()))
    return net(args)

"""
Get optimizer from str parameter
"""
def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    param_groups = net.parameters()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optimizer == "Rmsprop":
        optimizer = optim.RMSprop(param_groups,
                                  lr=args.lr,alpha=0.95,eps=1e-8)
    elif args.optimizer =="MarginRankingLoss":
       optimizer = MarginRankingLoss(margin=args.margin)
    else:
        raise ValueError('Not a valid optimizer')

   #  def poly_schd(epoch):
   #      return math.pow(1 - epoch / args.max_epoch, args.poly_exp)

   #  def poly2_schd(epoch):
   #      if epoch < args.poly_step:
   #          poly_exp = args.poly_exp
   #      else:
   #          poly_exp = 2 * args.poly_exp
   #      return math.pow(1 - epoch / args.max_epoch, poly_exp)
    
   #  if args.lr_schedule == 'poly2':
   #      scheduler = optim.lr_scheduler.LambdaLR(optimizer,
   #                                              lr_lambda=poly2_schd)
   #  elif args.lr_schedule == 'poly':
   #      scheduler = optim.lr_scheduler.LambdaLR(optimizer,
   #                                              lr_lambda=poly_schd)
   #  else:
   #      raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer #, scheduler


"""
Get activation function from str parameter
"""
def get_act_func(act_type):

 if act_type=="relu":
    return nn.ReLU()
            
 if act_type=="tanh":
    return nn.Tanh()
            
 if act_type=="sigmoid":
    return nn.Sigmoid()
           
 if act_type=="leaky_relu":
    return nn.LeakyReLU(0.2)
            
 if act_type=="elu":
    return nn.ELU()
                     
 if act_type=="softmax":
    return nn.LogSoftmax(dim=1)
        
 if act_type=="linear":
    return nn.LeakyReLU(1) # initializzed like this, but not used in forward!
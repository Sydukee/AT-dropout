import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=105, help='how many epochs to train')
parser.add_argument('--gpu', '-g', default=0, help='which gpu to use')
parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size')
# parser.add_argument('--auto-continue', default=False, type=bool, help='load last checkpoint and continue training')
parser.add_argument('--model-dir', default='./log/models', help='save path of models')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-adv_coef', default=1.0, type=float, help='Specify the weight for adversarial loss')
parser.add_argument('--val_interval', default=1, type=int, help='output information')
parser.add_argument('--val-method', '-vm', default='pgd', type=str, help='choose attack method')
parser.add_argument('--drop-prob', default=0, type=float, help='drop attack probability')
parser.add_argument('--lr', default=0.1)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight-decay', default=5e-4)
parser.add_argument('--at', default=True)
parser.add_argument('--net', default='resnet')

parser.add_argument('--class-num', default=10, type=int, help='how many classes to classify')
parser.add_argument('--epsilon', default=0.1, type=float, help='edge check range')
parser.add_argument('--theta', default=0.05, type=float, help='label adjustment')
parser.add_argument('--alpha', default=0.01, type=float, help='penal item weight')

args = parser.parse_args()
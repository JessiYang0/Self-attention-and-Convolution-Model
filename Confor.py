
import math
import Deep_tool
import torch
import argparse
import Wind
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from Deep_tool import MSE
from Deep_tool import score

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
import torch.nn.functional as F


from torch.autograd import Variable

torch.manual_seed(1111) 

parser = argparse.ArgumentParser(description='Sequence CONATT Modeling - Solar Energy Predict')

parser.add_argument('--model_name', type=str,default='CONATT',
                    help='Model name')
parser.add_argument('--plot_show',action='store_true',
                    help='If show plot')
parser.add_argument('--plot_save',action='store_false',
                    help='If save plot')
parser.add_argument('--run_number',type=int,default=1,
                    help='Number of experiment we want to run')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip, -1 means no clip (default: 1.0)')
parser.add_argument('--layer_dim', type=int,default=6,
                    help='number of layer')

parser.add_argument('--epochs', type=int, default=10000,
                    help='upper epoch limit (default: 1000)')

parser.add_argument('--early_stop', type=int,default=1000,
                    help='early stop after x epoch')

parser.add_argument('--n_heads', type=int, default=6,
                    help='attention head (default: 8)')
parser.add_argument('--n_layers', type=int, default=8,
                    help='attention head (default: 5)')

parser.add_argument('--seq_len', type=int, default=144,
                    help='initial history size (default: 144)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval (default: 50')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: RMSprop)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer (default: 10)')

parser.add_argument('--resample',action='store_false',
                    help='If resample')
parser.add_argument('--wavelet',action='store_true',
                    help='If wavelet')

parser.add_argument('--resam_num',type=int,default=5,
                    help='Resample per X minutes.')


args = parser.parse_args([])

model_name  = args.model_name
wavelet     = args.wavelet
resam       = args.resample
resam_num   = args.resam_num
hidd        = args.nhid
layer_dim   = args.layer_dim

#資料前處理
data,target = Wind.get_data(wavelet=wavelet,dummy=False)





seq_len = args.seq_len   
epochs = args.epochs
batch_size = args.batch_size
early_stop = args.early_stop
n_heads    = args.n_heads
n_layers   = args.n_layers

#切分訓練集、測試集(此validation為最終評分之資料
# ，訓練中用於驗證模型之資料會於創建dataset時切分)
train = data[:'2016-10-31']
validation = data['2016-11-01':] # is 365 days
train.shape, validation.shape

ytrain = target[:'2016-10-31']
yvalid = target['2016-11-01':] # is 365 days
ytrain.shape, yvalid.shape





#將2維資料轉換為3維資料
Group,t_group = Wind.transfer_3D( train,ytrain,seq_len)
t_group = t_group.reshape(t_group.shape[0],-1)  

time_index = validation.index

#創建dataset、dataloader同時切分訓練集以及驗證集
train_ds,valid_ds,_ = Deep_tool.create_datasets(Group,t_group)
train_dl,valid_dl   = Deep_tool.create_loaders(train_ds,valid_ds)



input_dim  = Group.shape[2]
output_dim = 1





#啟動函數

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()




#注意力功能

#先在外部，透過3個linear層將輸入轉換成Q、K、V，(多頭透過將維度乘以頭數n_heads
# 來達成，深入了解可看李宏毅教授的Transformer介紹影片，很清楚)
#之後再輸入至ScaledDotProductAttention進行注意力的計算
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k * n_heads)
        self.WK = nn.Linear(d_model, d_k * n_heads)
        self.WV = nn.Linear(d_model, d_v * n_heads)

        self.linear = nn.Linear(n_heads * d_v, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)


        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = ScaledDotProductAttention(d_k=self.d_k)(
            Q=q_s, K=k_s, V=v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + Q), attn

#多頭注意力機制內部計算注意力之Class
class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn





#注意力模塊中其他線性層、啟動函數、還有殘差功能的應用
class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)


#在我的方法中把一個MultiHeadAttention+一個PoswiseFeedForwardNet
#合併稱為一個注意力模塊。
class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(
                Q=enc_inputs, K=enc_inputs,
                V=enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

#堆疊多層(n_layers)注意力模塊來為框架應用注意力機制，層數為超參數
class MATT(nn.Module):

    def __init__(self, input_dim):
        super().__init__()



        self.layers = []
        
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=input_dim, d_ff=input_dim*2,
                d_k=input_dim*2, d_v=input_dim*4, n_heads=n_heads
                )
            self.layers.append(encoder_layer)

        self.layers = nn.ModuleList(self.layers)        

  
    
    def forward(self, x):

        for layer in self.layers:
            out, enc_self_attn = layer(x)
 
        return out
        



#卷積模塊
class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

#填充功能，讓經過卷積模塊前後的輸入輸出大小一致
def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)



class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 3,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)





#前饋模塊
class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)




class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)



#整體架構
class CAA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 3,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = MATT(input_dim)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

        self.fff = nn.Linear(dim, 1)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        x = self.fff(x)
        return x[:,:,-1]



model = CAA(dim = 6)
model = model.cuda()


lr = args.lr
i_epoch = len(train_dl)


#此function內部將整個訓練過程寫在內，整個計算損失以及反向傳播全部都在內，每一個epoch會執行一次
def train(epoch):
    global lr
    model.train()
#損失函數
    criterion = nn.MSELoss(size_average=True)
#最佳化器
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
#隨時間變化學習率功能，寫在Deep_tool.py內(cyclic learning rate)
    sched = Deep_tool.CyclicLR(opt,Deep_tool.cosine(t_max=i_epoch * 2,eta_min=lr/100))

    total_loss = 0

#使用批訓練，dataloader會讓train_dl此變數可跌代，使用for迴圈一批一批的將資料取出，損失計算以及更新參數也會在此for迴圈內執行
    for step,(x_batch,y_batch) in enumerate(train_dl):
#將資料使用GPU跑
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        sched.step()
        opt.zero_grad()
        
        out = model(x_batch)
        
        loss = criterion(out, y_batch.float())
        loss.backward()
#梯度裁減，減緩梯度爆炸影響
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        
        total_loss += loss.item()
        
        return total_loss

#此function內部將整個驗證過程寫在內，此部分資料為驗證集，不拿來更新參數，只用於判斷Early stopping以及測試模型性能
def evaluate(epoch):
    model.eval()
#不track梯度
    with torch.no_grad():
        valid   = valid_dl.dataset.tensors[0].cuda()
        valid_y = valid_dl.dataset.tensors[1].cuda()
        output = model(valid)
#每100epoch打印出Loss來看訓練狀況
        test_loss = MSE(output,valid_y)
        if epoch % 100 ==0:
            print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
    
        return test_loss.item()
    


    
best_loss = 999999999

losses  = []
vlosses = []

epochs = args.epochs

patience = 0


for epoch in range(epochs):
    train_error = train(epoch)
    test_error  = evaluate(epoch)
    losses.append(train_error)   
    vlosses.append(test_error)
    patience += 1
    if best_loss > test_error:
        best_loss = test_error
        torch.save(model.state_dict(),'best.pth')
        patience = 0 
    if patience > early_stop:
        break






#use model 
final_model = CAA(dim=6)
final_model.load_state_dict(torch.load('best.pth'))
final_model = final_model.cuda()


vGroup,t_vgroup = Wind.transfer_3D( validation,yvalid,time_step=seq_len)
vGroup = torch.Tensor(vGroup).cuda()
with torch.no_grad():
    out = final_model(vGroup)
    
    
out = out.cpu().numpy()   
t_vgroup = t_vgroup.reshape(t_vgroup.shape[0],-1)


Deep_tool.score(out,t_vgroup)




#run_number: 決定跑幾次模型
#save : 是否同時儲存設定好的圖片
def episode(run_number,save):
    
    global model_name,vGroup,t_vgroup,epochs
    
    episode_losses  = []
    episode_vlosses = []
    best_loss = 999999999
    for epoch in range(epochs):
        train_error = train(epoch)
        test_error  = evaluate(epoch)
        episode_losses.append(train_error)   
        episode_vlosses.append(test_error)
        
        if best_loss > test_error:
            torch.save(model.state_dict(),'Model_Save/LSTM/'+model_name+'_'+str(run_number)+'_model.pth')  
        
    
    if save == True:
        model.load_state_dict(torch.load('Model_Save/LSTM/'+model_name+'_'+str(run_number)+'_model.pth'))
        with torch.no_grad():
            out = model(vGroup.cuda())
    
        out = out.cpu().numpy()   
        t_vgroup = t_vgroup.reshape(t_vgroup.shape[0],-1)
        
        
        training_plt = plt.figure()
        plt.plot(episode_vlosses,label = 'Valid loss')
        plt.plot(episode_losses ,label = 'Train lss')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.legend()
        training_plt.savefig('Model_Save/LSTM/'+model_name+'_training_plt_'+str(run_number)+'_png')
        
            
        tf = plt.figure()
        plt.plot(out[1], label='predicted_power')
        plt.plot(t_vgroup[1], label='true_power')
        plt.xticks([0,20,40,60,80,100,120,140],
               time_index.hour[[0,20,40,60,80,100,120,140]])
        plt.legend()
        tf.savefig('Model_Save/LSTM/'+model_name+'_tf_'+str(run_number)+'_png')
        
        
        total_plt = plt.figure()     
        plt.plot(t_vgroup.flatten(), label='true_power')
        plt.plot(out.flatten(), label='predicted_power')
        plt.xticks([0,1000,2000,3000,4000,5000],
               time_index.strftime('%Y-%m-%d')[[[0,1000,2000,3000,4000,5000]]],
               rotation=33)
        plt.legend()
        total_plt.savefig('Model_Save/LSTM/'+model_name+'_total_plt_'+str(run_number)+'_png')

        
        nine_day = plt.figure()
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.plot(out[i], label='predicted_power')
            plt.plot(t_vgroup[i], label='true_power')
            plt.xticks([])
            plt.yticks([])
        nine_day.savefig('Model_Save/LSTM/'+model_name+'_9day_'+str(run_number)+'_png')
        
        plt.close('all')

    return episode_losses,episode_vlosses



plot_save = args.plot_save
run_number = args.run_number

vGroup,t_vgroup = Wind.transfer_3D( validation,yvalid,time_step=seq_len)
vGroup = torch.Tensor(vGroup)


total_loss = pd.DataFrame()

for i in range(run_number):
    model = LSTM(input_dim, hidd,layer_dim,output_dim)
    model = model.cuda()

    criterion = nn.MSELoss(size_average=True)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    sched = Deep_tool.CyclicLR(opt,Deep_tool.cosine(t_max=i_epoch * 2,eta_min=lr/100))

    


total_loss.to_csv('Model_Save/LSTM/'+model_name+'_total_loss.csv')





        
final_model = CAA(dim=6)
final_model.load_state_dict(torch.load('best.pth'))
final_model = final_model.cuda()


with torch.no_grad():
    out = final_model(vGroup)
    
    
out = out.cpu().numpy()   
t_vgroup = t_vgroup.reshape(t_vgroup.shape[0],-1)


Deep_tool.score(out,t_vgroup)



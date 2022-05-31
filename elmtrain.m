function model = elmtrain(P,T,N,TF,TYPE)
% ELMTRAIN 创建和训练一个极限学习机
% 语法
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% 描述
% 输入
% P - 训练集的输入矩阵(R*Q)
% T - 训练集的输出矩阵(S*Q)
% N - 隐性神经元的数量（默认=Q）。
% TF - 传递函数。
% 'sig' 表示正弦函数（默认）。
% 'sin' 代表正弦函数
% 'hardlim' 代表Hardlim函数
% TYPE - 回归（0，默认）或分类（1）。
% 输出
% IW - 输入权重矩阵(N*R)
% B - 偏置矩阵(N*1)
% LW - 层权重矩阵(N*S)
% 例子
% 回归。
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% 分类
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20, 'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
if nargin < 2
    error('ELM:Arguments','Not enough input arguments.');
end
if nargin < 3
    N = size(P,2);
end
if nargin < 4
    TF = 'sig';
end
if nargin < 5
    TYPE = 1;
end
if size(P,2) ~= size(T,2)
    error('ELM:Arguments','The columns of P and T must be same.');
end
[R,Q] = size(P);
if TYPE  == 1
    T=ind2vec(T);
end
[S,Q]=size(T);
% 随机生成输入权重矩阵
IW=rand(N,R) * 2 - 1;
% 随机生成偏置矩阵
B=rand(N,1);
BiasMatrix=repmat(B,1,Q);
% 计算图层输出矩阵H
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% 计算输出权重矩阵
LW = pinv(H') * T';
model.IW=IW;
model.B=B;
model.LW=LW;
model.TF=TF;
model.TYPE=TYPE;


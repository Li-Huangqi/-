function Y = elmpredict(P,model)
% ELMPREDICT 模拟一个极端学习机器
% 语法
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% 描述
% 输入
% P - 训练集的输入矩阵(R*Q)
% IW - 输入权重矩阵(N*R)
% B - 偏置矩阵(N*1)
% LW - 层权重矩阵(N*S)
% TF - 传递函数。
% 'sig'代表正弦函数（默认）。
% 'sin' 代表正弦函数
% 'hardlim'为Hardlim函数
% TYPE - 回归（0，默认）或分类（1）。
% 输出
% Y - 模拟输出矩阵 (S*Q)
% 例子
% 回归。
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% 分类

% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20, 'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
IW=model.IW;
B=model.B;
LW=model.LW;
TF=model.TF;
TYPE=model.TYPE;
% 计算图层输出矩阵H
Q = size(P,2);
BiasMatrix = repmat(B,1,Q);
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% 计算模拟输出
Y = (H' * LW)';
if TYPE == 1
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
        [max_Y,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); 
end
       
    

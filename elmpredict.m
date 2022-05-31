function Y = elmpredict(P,model)
% ELMPREDICT ģ��һ������ѧϰ����
% �﷨
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% ����
% ����
% P - ѵ�������������(R*Q)
% IW - ����Ȩ�ؾ���(N*R)
% B - ƫ�þ���(N*1)
% LW - ��Ȩ�ؾ���(N*S)
% TF - ���ݺ�����
% 'sig'�������Һ�����Ĭ�ϣ���
% 'sin' �������Һ���
% 'hardlim'ΪHardlim����
% TYPE - �ع飨0��Ĭ�ϣ�����ࣨ1����
% ���
% Y - ģ��������� (S*Q)
% ����
% �ع顣
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% ����

% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20, 'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
IW=model.IW;
B=model.B;
LW=model.LW;
TF=model.TF;
TYPE=model.TYPE;
% ����ͼ���������H
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
% ����ģ�����
Y = (H' * LW)';
if TYPE == 1
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
        [max_Y,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); 
end
       
    

function model = elmtrain(P,T,N,TF,TYPE)
% ELMTRAIN ������ѵ��һ������ѧϰ��
% �﷨
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% ����
% ����
% P - ѵ�������������(R*Q)
% T - ѵ�������������(S*Q)
% N - ������Ԫ��������Ĭ��=Q����
% TF - ���ݺ�����
% 'sig' ��ʾ���Һ�����Ĭ�ϣ���
% 'sin' �������Һ���
% 'hardlim' ����Hardlim����
% TYPE - �ع飨0��Ĭ�ϣ�����ࣨ1����
% ���
% IW - ����Ȩ�ؾ���(N*R)
% B - ƫ�þ���(N*1)
% LW - ��Ȩ�ؾ���(N*S)
% ����
% �ع顣
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% ����
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
% �����������Ȩ�ؾ���
IW=rand(N,R) * 2 - 1;
% �������ƫ�þ���
B=rand(N,1);
BiasMatrix=repmat(B,1,Q);
% ����ͼ���������H
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% �������Ȩ�ؾ���
LW = pinv(H') * T';
model.IW=IW;
model.B=B;
model.LW=LW;
model.TF=TF;
model.TYPE=TYPE;


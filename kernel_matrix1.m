function omega = kernel_matrix1(Xtrain,kernel_pars,Xtest)
% ����:Xtrainÿ��Ϊһ������
if nargin<3
        XXh = sum(Xtrain.^2,2)*ones(1,size(Xtrain,1));
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars);
    else
    %�������4ʱ�ǽ���������ӳ�䵽�˿ռ䣬��ʱ��һ���������Ϊѵ������
    %��4������Ϊ��������
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xtest,1));
        XXh2 = sum(Xtest.^2,2)*ones(1,size(Xtrain,1));
        omega = XXh1+XXh2' - 2*Xtrain*Xtest';
        omega = exp(-omega./kernel_pars);
end
%   if nargin<3
%         XXh = sum(Xtrain.^2,2)*ones(1,size(Xtrain,1));
%         omega = XXh+XXh'-2*(Xtrain*Xtrain');
%         XXh1 = sum(Xtrain,2)*ones(1,size(Xtrain,1));
%         omega1 = XXh1-XXh1';
%         omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
%     else
%         XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xtest,1));
%         XXh2 = sum(Xtest.^2,2)*ones(1,size(Xtrain,1));
%         omega = XXh1+XXh2' - 2*(Xtrain*Xtest');
%         XXh11 = sum(Xtrain,2)*ones(1,size(Xtest,1));
%         XXh22 = sum(Xtest,2)*ones(1,size(Xtrain,1));
%         omega1 = XXh11-XXh22';
%         omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
%     end
end
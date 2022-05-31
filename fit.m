function fitness=fit(train_data,train_data_label,RegularCoef,KernelArgs,V)
%����Ϊ���ý�����֤��ѵ�����Ͳ��Լ�
% ��������
data=[train_data,train_data_label];
[data_r, data_c] = size(data);
%��������������ָ�Ϊ5����
indices = crossvalind('Kfold',data_r,V);
for i=1:V
    % ��ȡ��i�ݲ������ݵ������߼�ֵ
    test = (indices == i); 
    % ȡ������ȡ��i��ѵ�����ݵ������߼�ֵ
    train = ~test;
    %1�ݲ��ԣ�4��ѵ��
    test_data=data(test, 1 : data_c - 1);
    test_data_label=data(test, data_c);
    train_data = data(train, 1 : data_c - 1);
    train_data_label=data(train, data_c);
    % ʹ�����ݵĴ���
    model=kelmtrain(train_data,train_data_label,RegularCoef,KernelArgs);
    [~, ~,TestingAccuracy] = kelmpredict(test_data,test_data_label,model); 
    fitness(i)=TestingAccuracy;
end
fitness=sum(fitness)/i;

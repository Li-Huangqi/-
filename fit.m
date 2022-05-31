function fitness=fit(train_data,train_data_label,RegularCoef,KernelArgs,V)
%函数为设置交叉验证的训练集和测试集
% 导入数据
data=[train_data,train_data_label];
[data_r, data_c] = size(data);
%将数据样本随机分割为5部分
indices = crossvalind('Kfold',data_r,V);
for i=1:V
    % 获取第i份测试数据的索引逻辑值
    test = (indices == i); 
    % 取反，获取第i份训练数据的索引逻辑值
    train = ~test;
    %1份测试，4份训练
    test_data=data(test, 1 : data_c - 1);
    test_data_label=data(test, data_c);
    train_data = data(train, 1 : data_c - 1);
    train_data_label=data(train, data_c);
    % 使用数据的代码
    model=kelmtrain(train_data,train_data_label,RegularCoef,KernelArgs);
    [~, ~,TestingAccuracy] = kelmpredict(test_data,test_data_label,model); 
    fitness(i)=TestingAccuracy;
end
fitness=sum(fitness)/i;

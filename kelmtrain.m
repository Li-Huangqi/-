function model=kelmtrain(train_data,train_data_label,RegularCoef,KernelArgs)
train_data=train_data';
train_data_label=train_data_label';
C = RegularCoef;
NumberofTrainingData=size(train_data,2);
sorted_target=sort(cat(2,train_data_label),2);
    label=zeros(1,1);                               %从训练和测试数据集中找到并保存在'标签'类标签中
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    %%%%%%%%%% 处理培训的目标
    temp_T=zeros(NumberofOutputNeurons,NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == train_data_label(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    train_data_label=temp_T*2-1;
    %%%%%%%%%% 处理测试的目标
n = size(train_data_label,2);
Omega_train = kernel_matrix1(train_data', KernelArgs);
OutputWeight=((Omega_train+speye(n)/C)\(train_data_label')); 
Y=(Omega_train * OutputWeight)';                             % Y：训练数据的实际输出
%%%%%%%%%% 计算训练和测试分类的准确性
Actual_label=[];
expected_label=[];
MissClassificationRate_Training=0; 
    for i = 1 : size(train_data_label, 2)
        [~,label_index_actual]=max(train_data_label(:,i));
        [~,label_index_expected]=max(Y(:,i));
        expected_label(i)=label_index_expected;
        Actual_label(i)=label_index_actual;
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainAccuracy=1-MissClassificationRate_Training/size(train_data_label,2); 
    model.train=train_data;
    model.OutputWeight=OutputWeight;
    model.expected_label=expected_label;
    model.Actual_label=Actual_label;
    model.TrainAccuracy=TrainAccuracy;
    model.RegularCoef=RegularCoef;
    model.KernelArgs=KernelArgs;
end
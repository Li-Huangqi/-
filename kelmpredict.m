function [expected_label,Actual_label,TestingAccuracy] = kelmpredict(test_data,test_data_label,model) 
    train_data=model.train;
    OutputWeight=model.OutputWeight;
    RegularCoef=model.RegularCoef;
    KernelArgs=model.KernelArgs;
test_data=test_data';
test_data_label=test_data_label';
C = RegularCoef;
NumberofTestingData=size(test_data,2);
sorted_target=sort(cat(2,test_data_label),2);
    label=zeros(1,1);                               %从训练和测试数据集中找到并保存在'标签'类标签中
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == test_data_label(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    test_data_label=temp_TV_T*2-1;                           %结束时，Elm_Type的百分比
%%%%%%%%%%% 计算测试输入的输出
Omega_test = kernel_matrix1(train_data',KernelArgs,test_data');
TY=(Omega_test' * OutputWeight)';                            %TY：测试数据的实际输出。
%%%%%%%%%% 计算训练和测试分类的准确性
Actual_label=[];
expected_label=[];
MissClassificationRate_Testing=0; 
    for i = 1 : size(test_data_label, 2)
        [~,label_index_actual]=max(test_data_label(:,i));
        [~,label_index_expected]=max(TY(:,i));
        expected_label(i)=label_index_expected;
        Actual_label(i)=label_index_actual;
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(test_data_label,2);  
end
function [bestc,bestg,Convergence_curve]=IGWO(train_data_label,train_data,str)
SearchAgents_no=str.sizepop;
Max_iteration=str.maxgen;
lb=[str.popcmin,str.popgmin];
ub=[str.popcmax,str.popgmax];
v = str.v; % SVM Cross Validation参数,默认为5
dim=2; % 此例需要优化,两个参数c和g
% 初始化alpha, beta, and delta
Alpha_pos=zeros(1,dim); % 初始化Alpha狼的位置
Alpha_score=inf; % 初始化Alpha狼的目标函数值
Beta_pos=zeros(1,dim); % 初始化Beta狼的位置
Beta_score=inf; % 初始化Beta狼的目标函数值
Delta_pos=zeros(1,dim); % 初始化Delta狼的位置
Delta_score=inf; % 初始化Delta狼的目标函数值
%初始化搜索狼的位置
for i=1:dim
    ub_i=ub(i);
    lb_i=lb(i);
    Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
end
Convergence_curve=zeros(1,Max_iteration);
velocity =0.3*randn(SearchAgents_no,dim) ;
% w=0.5+rand()/2;
l=0; % Loop counter循环计数器
% Main loop主循环
while l<Max_iteration  % 对迭代次数循环
    for i=1:size(Positions,1)  % 遍历每个狼
       % 若搜索位置超过了搜索空间，需要重新回到搜索空间
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        % 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界；
        % 若超出最小值，最回答最小值边界
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % ~表示取反          
      % 计算适应度函数值
       cmd = ['-v ',num2str(v),' -c ',num2str(Positions(i,1)),' -g ',num2str(Positions(i,2))];
       fitness(i,1)=svmtrain(train_data_label,train_data,cmd); % SVM模型训练
       fitness=-fitness(i); 
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score % 如果目标函数值小于Alpha狼的目标函数值
            Alpha_score=fitness; % 则将Alpha狼的目标函数值更新为最优目标函数值，Update alpha
            Alpha_pos=Positions(i,:); % 同时将Alpha狼的位置更新为最优位置
        end
        if fitness>Alpha_score && fitness<Beta_score % 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
            Beta_score=fitness; % 则将Beta狼的目标函数值更新为最优目标函数值，Update beta
            Beta_pos=Positions(i,:); % 同时更新Beta狼的位置
        end
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score  % 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
            Delta_score=fitness; % 则将Delta狼的目标函数值更新为最优目标函数值
            Delta_pos=Positions(i,:); % 同时更新Delta狼的位置
        end
    end
% a=2-(l*(2)/Max_iteration); % 对每一次迭代，计算相应的a值
   w=pi*((l/Max_iteration).^2)-pi*(l/Max_iteration)+1;
%  a=2*cos(pi/2*l/Max_iteration); % 对每一次迭代，计算相应的a值，
%     a=2-2*((exp(l/Max_iteration)-1)/exp(1)-1);
%     a=tanh(-2*pi*l/Max_iteration+pi)+1;
    a=2*sqrt(1-(l/Max_iteration)^2);
    %更新包括Omegas在内的狼的位置
    for i=1:size(Positions,1) % 遍历每个狼
        for j=1:size(Positions,2) % 遍历每个维度
            % 包围猎物，位置更新
            r1=rand(); % r1是[0,1]中的一个随机数
            r2=rand(); % r2是[0,1]中的一个随机数
            A1=2*a*r1-a; % 计算系数A
            C1=2*r2; % 计算系数C
            % Alpha狼位置更新
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); 
            x1=Alpha_pos(j)-A1*D_alpha; 
            r1=rand();
            r2=rand(); 
            A2=2*a*r1-a; 
            C2=2*r2; 
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j));
            x2=Beta_pos(j)-A2*D_beta;  
            r1=rand();
            r2=rand();
            A3=2*a*r1-a; 
            C3=2*r2;
            % Delta狼位置更新
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); 
            x3=Delta_pos(j)-A3*D_delta;  
            % 位置更新
            r1=rand();
            r2=rand();
            r3=rand();
            velocity(i,j)=w*(velocity(i,j)+C1*r1*(x1-Positions(i,j))+C2*r2*(x2-Positions(i,j))+C3*r3*(x3-Positions(i,j)));
            Positions(i,j)=Positions(i,j)+velocity(i,j);
        end
    end
    l=l+1;   
    Convergence_curve(l)=Alpha_score;
    bestc=Alpha_pos(1,1);
bestg=Alpha_pos(1,2);
end
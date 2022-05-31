function [Convergence_curve,bestc,bestg]=IGWO_KELM(train_data,train_data_label,str)
SearchAgents_no=str.sizepop;
Max_iteration=str.maxgen;
lb=[str.popcmin,str.popgmin];
ub=[str.popcmax,str.popgmax];
V= str.v; % SVM Cross Validation参数,默认为5
dim=2;
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
while l<Max_iteration  % 对迭代次数循环
    for i=1:size(Positions,1) % 遍历每个狼
       % Return back the search agents that go beyond the boundaries of the search space
       % 若搜索位置超过了搜索空间，需要重新回到搜索空间
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        % 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界；
        % 若超出最小值，最回答最小值边界
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % ~表示取反          
      % 计算适应度函数值
         RegularCoef=Positions(i,1);
         KernelArgs=Positions(i,2);
         fitness=fit(train_data,train_data_label,RegularCoef,KernelArgs,V);
         fitness=-fitness; %以错误率最小化为目标
        % Update Alpha, Beta, and Delta
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
            Delta_score=fitness; % 则将Delta狼的目标函数值更新为最优目标函数值，Update delta
            Delta_pos=Positions(i,:); % 同时更新Delta狼的位置
        end
    end
%    a=2-(l*(2)/Max_iteration); % 对每一次迭代，计算相应的a值
   w=pi*((l/Max_iteration).^2)-pi*(l/Max_iteration)+1;
   a=2*cos(pi/2*l/Max_iteration); % 对每一次迭代，计算相应的a值，
%     a=2-2*((exp(l/Max_iteration)-1)/exp(1)-1);
%    a=tanh(-2*pi*l/Max_iteration+pi)+1;    %更新包括Omegas在内的狼的位置
    for i=1:size(Positions,1) % 遍历每个狼
        for j=1:size(Positions,2) % 遍历每个维度
            % 包围猎物，位置更新
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            A1=2*a*r1-a; % 计算系数A，Equation (3.3)
            C1=2*r2; % 计算系数C，Equation (3.4)
            % Alpha狼位置更新
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            x1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
            r1=rand();
            r2=rand(); 
            A2=2*a*r1-a; % 计算系数A，Equation (3.3)
            C2=2*r2; % 计算系数C，Equation (3.4)
            % Beta狼位置更新
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            x2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2      
            r1=rand();
            r2=rand();
            A3=2*a*r1-a; % 计算系数A，Equation (3.3)
            C3=2*r2; % 计算系数C，Equation (3.4)
            % Delta狼位置更新
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            x3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3            
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
end
bestc=Alpha_pos(1,1);
bestg=Alpha_pos(1,2);

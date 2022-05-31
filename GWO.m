function [bestc,bestg,Convergence_curve]=GWO(train_data_label,train_data,str)
SearchAgents_no=str.sizepop;
Max_iteration=str.maxgen;
lb=[str.popcmin,str.popgmin];
ub=[str.popcmax,str.popgmax];
v = str.v; % SVM Cross Validation����,Ĭ��Ϊ5
dim=2; % ������Ҫ�Ż�,��������c��g% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim); % ��ʼ��Alpha�ǵ�λ��
Alpha_score=inf; % ��ʼ��Alpha�ǵ�Ŀ�꺯��ֵ
Beta_pos=zeros(1,dim); % ��ʼ��Beta�ǵ�λ��
Beta_score=inf; % ��ʼ��Beta�ǵ�Ŀ�꺯��ֵ
Delta_pos=zeros(1,dim); % ��ʼ��Delta�ǵ�λ��
Delta_score=inf; % ��ʼ��Delta�ǵ�Ŀ�꺯��ֵ
%��ʼ�������ǵ�λ��
for i=1:dim
    ub_i=ub(i);
    lb_i=lb(i);
    Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
end
Convergence_curve=zeros(1,Max_iteration);
l=0; % Loop counterѭ��������
% Main loop��ѭ��
while l<Max_iteration  % �Ե�������ѭ��
   for i=1:size(Positions,1)  % ����ÿ����
       % ������λ�ó����������ռ䣬��Ҫ���»ص������ռ�
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        % ���ǵ�λ�������ֵ����Сֵ֮�䣬��λ�ò���Ҫ���������������ֵ����ص����ֵ�߽磻
        % ��������Сֵ����ش���Сֵ�߽�
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % ~��ʾȡ��          
      % ������Ӧ�Ⱥ���ֵ
       cmd = ['-v ',num2str(v),' -c ',num2str(Positions(i,1)),' -g ',num2str(Positions(i,2))];
       fitness(i,1)=svmtrain(train_data_label,train_data,cmd); % SVMģ��ѵ��
       fitness=-fitness(i); % �Դ�������С��ΪĿ��
        if fitness<Alpha_score % ���Ŀ�꺯��ֵС��Alpha�ǵ�Ŀ�꺯��ֵ
            Alpha_score=fitness; % ��Alpha�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ��Update alpha
            Alpha_pos=Positions(i,:); % ͬʱ��Alpha�ǵ�λ�ø���Ϊ����λ��
        end
       if fitness>Alpha_score && fitness<Beta_score % ���Ŀ�꺯��ֵ������Alpha�Ǻ�Beta�ǵ�Ŀ�꺯��ֵ֮��
            Beta_score=fitness; % ��Beta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ��Update beta
            Beta_pos=Positions(i,:); % ͬʱ����Beta�ǵ�λ��
        end
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score  % ���Ŀ�꺯��ֵ������Beta�Ǻ�Delta�ǵ�Ŀ�꺯��ֵ֮��
            Delta_score=fitness; % ��Delta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Delta_pos=Positions(i,:); % ͬʱ����Delta�ǵ�λ��
        end
    end
    a=2-(l*(2)/Max_iteration); % ��ÿһ�ε�����������Ӧ��aֵ
    %���°���Omegas���ڵ��ǵ�λ��
    for i=1:size(Positions,1) % ����ÿ����
        for j=1:size(Positions,2) % ����ÿ��ά��
            % ��Χ���λ�ø���
            r1=rand(); % r1��[0,1]�е�һ�������
            r2=rand(); % r2��[0,1]�е�һ�������
            A1=2*a*r1-a; % ����ϵ��A
            C1=2*r2; % ����ϵ��C
            % Alpha��λ�ø���
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
            % Delta��λ�ø���
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); 
            x3=Delta_pos(j)-A3*D_delta;          
            % λ�ø���
            Positions(i,j)=(x1+x2+x3)/3;
        end
    end
    l=l+1;   
    Convergence_curve(l)=Alpha_score;
    bestc=Alpha_pos(1,1);
bestg=Alpha_pos(1,2);
end
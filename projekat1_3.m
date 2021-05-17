clc, clear, close all

%% Ucitavanje podataka
data = readtable('Nursery_dataset.csv','ReadVariableNames',0);
data1=table2array(data);
data3=transpose(data1);
data3=data3(:,2:10552);
data2=zeros(9,10551);

% pretvaranje stringova u brojeve
for i = 1:1:10551
        if(strcmp(data3(1,i),'usual')==true) 
            data2(1,i) = 0; end
        if(strcmp(data3(1,i),'pretentious')==true) 
            data2(1,i)= 1; end
        if(strcmp(data3(1,i),'great_pret')==true) 
            data2(1,i) = 2; end
        if(strcmp(data3(2,i),'proper')==true) 
            data2(2,i) = 0; end
        if(strcmp(data3(2,i),'less_proper')==true) 
            data2(2,i) = 1; end
        if(strcmp(data3(2,i),'improper')==true) 
            data2(2,i) = 2; end
        if(strcmp(data3(2,i),'critical')==true) 
            data2(2,i) = 3; end
        if(strcmp(data3(2,i),'very_crit')==true) 
            data2(2,i) = 4; end
        if(strcmp(data3(3,i),'complete')==true) 
            data2(3,i) = 0; end
        if(strcmp(data3(3,i),'completed')==true) 
            data2(3,i) = 1; end
        if(strcmp(data3(3,i),'incomplete')==true) 
            data2(3,i) = 2; end
        if(strcmp(data3(3,i),'foster')==true) 
            data2(3,i) = 3; end
        if(strcmp(data3(4,i),'1')==true) 
            data2(4,i) = 0; end
        if(strcmp(data3(4,i),'2')==true) 
            data2(4,i) = 1; end
        if(strcmp(data3(4,i),'3')==true) 
            data2(4,i) = 2; end
        if(strcmp(data3(4,i),'more')==true) 
            data2(4,i) = 3; end
        if(strcmp(data3(5,i),'convenient')==true) 
            data2(5,i) = 0; end
        if(strcmp(data3(5,i),'less_conv')==true) 
            data2(5,i) = 1; end
        if(strcmp(data3(5,i),'critical')==true) 
            data2(5,i) = 2; end
        if(strcmp(data3(6,i),'convenient')==true) 
            data2(6,i) = 0; end
        if(strcmp(data3(6,i),'inconv')==true) 
            data2(6,i) = 1; end
        if(strcmp(data3(7,i),'non-prob')==true) 
            data2(7,i) = 0; end
        if(strcmp(data3(7,i),'slightly_prob')==true) 
            data2(7,i) = 1; end
        if(strcmp(data3(7,i),'problematic')==true) 
            data2(7,i) = 2; end
        if(strcmp(data3(8,i),'recommended')==true) 
            data2(8,i) = 0; end
        if(strcmp(data3(8,i),'priority')==true) 
            data2(8,i) = 1; end
        if(strcmp(data3(8,i),'not_recom')==true) 
            data2(8,i) = 2; end
        if(strcmp(data3(9,i),'not_recom')==true) 
            data2(9,i) = 0; end
        if(strcmp(data3(9,i),'recommend')==true) 
            data2(9,i) = 1; end
        if(strcmp(data3(9,i),'very_recom')==true) 
            data2(9,i) = 2; end
        if(strcmp(data3(9,i),'priority')==true)
            data2(9,i) = 3; end
        if(strcmp(data3(9,i),'spec_prior')==true)
            data2(9,i) = 4; end
end

ulaz = data2(1:8,:);
izlaz = data2(9,:);

%% Odvajanje test i trening skupa

K0=data2(1:8,izlaz==0);
K1=data2(1:8,izlaz==1);
K2=data2(1:8,izlaz==2);
K3=data2(1:8,izlaz==3);
K4=data2(1:8,izlaz==4);

N0=length(K0);
N1=length(K1);
N2=length(K2);
N3=length(K3);
N4=length(K4);

K0Trening = K0(:,1:floor(0.9*N0));
K1Trening = K1(:,1:floor(0.9*N1));
K2Trening = K2(:,1:floor(0.9*N2));
K3Trening = K3(:,1:floor(0.9*N3));
K4Trening = K4(:,1:floor(0.9*N4));

K0Test = K0(:,floor(0.9*N0)+1:N0);
K1Test = K1(:,floor(0.9*N1)+1:N1);
K2Test = K2(:,floor(0.9*N2)+1:N2);
K3Test = K3(:,floor(0.9*N3)+1:N3);
K4Test = K4(:,floor(0.9*N4)+1:N4);

ulazTest=[K0Test, K1Test, K2Test, K3Test, K4Test];
nula=zeros(1,floor(N0*0.1));
jedan=ones(1,floor(N1*0.1));
dva=2*ones(1,floor(N2*0.1));
tri=3*ones(1,floor(N3*0.1));
cetiri=4*ones(1,floor(N4*0.1));
izlazTest=[nula, jedan, dva, tri, cetiri];

ind_test=randperm(length(izlazTest));
ulazTest=ulazTest(:,ind_test);
izlazTest=izlazTest(ind_test);

ulazTrening=[K0Trening, K1Trening, K2Trening, K3Trening, K4Trening];
nula=zeros(1,floor(N0*0.9));
jedan=ones(1,floor(N1*0.9));
dva=2*ones(1,floor(N2*0.9));
tri=3*ones(1,floor(N3*0.9));
cetiri=4*ones(1,floor(N4*0.9));
izlazTrening=[nula, jedan, dva, tri, cetiri];

ind_trening=randperm(length(izlazTrening));
izlazTrening=izlazTrening(ind_trening);
ulazTrening=ulazTrening(:,ind_trening);

trening_length = [5,length(ulazTrening)];
test_length = [5,length(ulazTest)];
izlazTreningMoj = zeros(trening_length);
izlazTestMoj = zeros(test_length);
max_trening = max(size(izlazTrening));
max_test = max(size(izlazTest));


for i = 1:1:max_trening
    switch true
        case(izlazTrening(1,i)==0) 
            izlazTreningMoj(5,i) = 1;
        case(izlazTrening(1,i)==1) 
            izlazTreningMoj(4,i) = 1;
        case(izlazTrening(1,i)==2) 
            izlazTreningMoj(3,i) = 1;
        case(izlazTrening(1,i)==3) 
            izlazTreningMoj(2,i) = 1; 
        case(izlazTrening(1,i)==4) 
            izlazTreningMoj(1,i) = 1;
    end
end

for i = 1:1:max_test
    switch true
        case(izlazTest(1,i)==0) 
            izlazTestMoj(5,i) = 1;
        case(izlazTest(1,i)==1) 
            izlazTestMoj(4,i) = 1;
        case(izlazTest(1,i)==2) 
            izlazTestMoj(3,i) = 1;
        case(izlazTest(1,i)==3) 
            izlazTestMoj(2,i) = 1; 
        case(izlazTest(1,i)==4) 
            izlazTestMoj(1,i) = 1;
    end
end

%% Odbirci na trening skupu - prikaz
yTrenira = [max(size(K0Trening)),max(size(K1Trening)),max(size(K2Trening)),max(size(K3Trening)),max(size(K4Trening))];
xTrenira = 0:1:4;
figure('Name','Prikaz odbiraka na trening skupu'), hold all
bar(xTrenira,yTrenira);

%% Odbirci na test skupu - prikaz
yTestira = [max(size(K0Test)),max(size(K1Test)),max(size(K2Test)),max(size(K3Test)),max(size(K4Test))];
xTestira= 0:1:4;
figure('Name','Prikaz odbiraka  na test skupu'), hold all
bar(xTestira,yTestira);

%% Krosvalidacija
Ntrening = length(ulazTrening);
bestF1 = 0;
bestarh = [7 7 7 7];
bestActivation = 'poslin';
bestRegularization = 0.1;
bestEpochNum = 700;

ulazSve=[ulazTrening, ulazTest];
izlazSve=[izlazTrening, izlazTest];
XTest = ulazTrening(:, [ceil(0.8*Ntrening)+1:Ntrening]);
YTest = izlazTreningMoj(:,[ceil(0.8*Ntrening)+1:Ntrening]);


for arhitektura = {[5 5], [10 20 30], [25 25 25]}
    for weight=[2,4,6]
    % Kreiranje NM sa trenutnom kombinacijom hiperparametara
    net = patternnet(arhitektura{1});
    net.layers{length(arhitektura{1})+1}.transferFcn = 'softmax';
    net.layers{length(arhitektura{1})+1}.size=5;
    
   for fcn = {'tansig', 'logsig', 'poslin'}
       
         for i = 1:length(arhitektura{1})
            net.layers{i}.transferFcn = fcn{1};
         end
    
        for reg = 0:0.25:1
            % Definisanje koeficijenta regularizacije
            net.performParam.regularization = reg;
            
           % net.trainFcn='traingdm';
            
            % Aktiviranje zastite od preobucavanja na osnovu validacionog skupa
            net.divideFcn = 'divideind';
            
            % Podatke podeliti na osnovu indeksa, 90% train, 10% test, 0% test
            net.divideParam.trainInd = 1:floor(0.9*Ntrening);
            net.divideParam.testInd = [];
             net.divideParam.valInd = floor(0.9*Ntrening)+1:Ntrening;
            
            % Podesiti maksimalan broj epoha treniranja
            net.trainParam.epochs = 1000;
            
            % Podesiti na koliko epoha se gleda rast greske na val skupu
            net.trainParam.max_fail = 10;
            
            % Podesiti maksimalnu dozvoljenu gresku
            net.trainParam.goal = 10e-5;
            
            % Podesiti vrednost minimalnog gradijenta
            net.trainParam.min_grad = 10e-6;
            
            net.trainparam.showWindow = false;
            
            W = zeros(1,Ntrening);
            
            W(izlazTreningMoj==1)=weight;
            
            % Treniranje neuralne mreze nad trening skupom
            [net,tr] = train(net,ulazTrening , izlazTreningMoj, [], [], W);

            % Ispitivanje NM nad validacionim podacima
            predTest = net(XTest);
            
            % Racunanje konfuzione matrice
            [c, cm] = confusion(YTest, predTest);
            cm = cm';
            
            % Racunanje performansi
            sumDijagonal = 0;
            FP = 0; %False positive
            TN = 0; %True negative
            TP = 0; %True positive
            FN = 0; %False negative
            P = 0; 
            R = 0; 
            A = 0;
            avgP = 0; 
            avgR = 0; 
            avgA = 0;

            for i=1:5
                sumDijagonal = sumDijagonal + cm(i,i);
            end

            for i=1:5
                TP = cm(i,i);
                TN = sumDijagonal - TP;
                for j=1:5
                    if(i~=j)
                        FP = FP + cm(i,j);
                        FN = FN + cm(j,i);
                    end
                end
                P = TP/(TP+FP);
                R = TP/(TP+FN);
                A = (TP+TN)/(TP+TN+FP+FN);
                avgP = avgP + P;
                avgR = avgR + R;
                avgA = avgA + A;
                P = 0;
                R = 0; 
                A = 0;
            end

            P = avgP/5;
            R = avgR/5;
            A = avgA/5;

            if A > bestF1 & not(isnan(A))
                bestF1 = A;
                bestStruct = arhitektura{1};
                bestActivation = fcn{1};
                bestRegularization = reg;
                bestEpochNum = tr.best_epoch;
                bestWeight = weight;
            end
        end
   end
    end
    end
%%
Abest = bestF1
bestReg = bestRegularization
bestStructure = bestStruct
bestEpochNumber = bestEpochNum
bestActivationF = bestActivation
bestWeight_ = bestWeight

%% Treniranje sa najboljim hiperparametrima
net = patternnet(bestStruct);
for i = 1:1:length(arhitektura{1})
    net.layers{i}.transferFcn = bestActivation
end
net.layers{length(bestStruct)+1}.transferFcn = 'softmax';
net.layers{length(bestStruct)+1}.size = 5;
net.performParam.regularization = bestRegularization

% Aktiviranje zastite od preobucavanja na osnovu validacionog skupa
net.divideFcn = '';

net.trainParam.max_fail = 10;
net.trainParam.goal = 10e-6;
net.trainParam.min_grad = 10e-6;
net.trainParam.epochs = bestEpochNum
net.trainparam.showWindow = true;

net = train(net, ulazTrening, izlazTreningMoj);

%% Testiranje nad test podacima

Ntest = length(ulazTest);
izlazTrening_pred = net(ulazTrening);

figure
plotconfusion(izlazTreningMoj, izlazTrening_pred, 'Konfuziona matrica nad skupom trening');

[c, cm] = confusion(izlazTreningMoj, izlazTrening_pred);
cm = cm';

sumaDiagonal = 0;
FP = 0; 
TN = 0; 
TP = 0;
FN = 0; 
P = 0; 
R = 0; 
A = 0;
avgP = 0;
avgR = 0;
avgA = 0;

for i=1:5
    sumaDiagonal = sumaDiagonal + cm(i,i);
end

for i=1:5
    TP = cm(i,i);
    TN = sumaDiagonal - TP;
    for j=1:5
        if(i~=j)
            FP = FP + cm(i,j);
            FN = FN + cm(j,i);
        end
    end
    P = TP/(TP+FP);
    R = TP/(TP+FN);
    A = (TP+TN)/(TP+TN+FP+FN);
    avgP = avgP + P;
    avgR = avgR + R;
    avgA = avgA + A;
    P = 0;
    R = 0;
    A = 0;
end
Ptrening = avgP/5
Rtrening = avgR/5
Atrening = avgA/5

izlazTest_pred = net(ulazTest);

figure
plotconfusion(izlazTestMoj, izlazTest_pred,'Konfuziona matrica nad TEST skupom');

% Racunanje konfuzione matrice
[c, cm] = confusion(izlazTestMoj, izlazTest_pred);
cm = cm';
 
 % Racunanje pokazatelja performansi za drugu klasu
DijagonalaSuma = 0;
FP = 0; 
TN = 0; 
TP = 0; 
FN = 0; 
P = 0; 
R = 0; 
A = 0;
Pavg = 0; Ravg = 0; Aavg = 0;
for i=1:5
    DijagonalaSuma = DijagonalaSuma + cm(i,i);
end
for i=1:5
    TP = cm(i,i);
    TN = DijagonalaSuma - TP;
    for j=1:5
        if(i~=j)
            FP = FP + cm(i,j);
            FN = FN + cm(j,i);
        end
    end
    P = TP/(TP+FP);
    R = TP/(TP+FN);
    A = (TP+TN)/(TP+TN+FP+FN);
    avgP = avgP + P;
    avgR = avgR + R;
    avgA = avgA + A;
    P = 0;
    R = 0;
    A = 0;
end

Ptest = avgP/5
Rtest = avgR/5
Atest = avgA/5

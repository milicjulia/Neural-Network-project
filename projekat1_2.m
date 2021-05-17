%%Ucitavanje podataka!

clc,clear,close all 

load 'dataset2';

ob = pod(:, 1:2)';
klasa = pod(:, 3)';
N = length(klasa);

K1 = ob(:, klasa == 1);
K2 = ob(:, klasa == 2);
K3 = ob(:, klasa == 3);


%% Vizualizacija podataka
figure, hold all
plot(K1(1, :), K1(2, :), 'o')
plot(K2(1, :), K2(2, :), '*')
plot(K3(1, :), K3(2, :), 'd')

%% One-hot encoding
izlaz = zeros(3, N);

izlaz(1, klasa == 1) = 1;
izlaz(2, klasa == 2) = 1;
izlaz(3, klasa == 3) = 1;

ulaz = ob;


%% podela na trening i test skup(80:20)

ind = randperm(N);

indTrening = ind(1 : 0.8*N);
indTest = ind(0.8*N+1 : N);

ulazTrening = ulaz(:,indTrening);
izlazTrening = izlaz(:,indTrening);

ulazTest = ulaz(:, indTest);
izlazTest = izlaz(:, indTest);

%% Mreza koja na optimalan nacin klasifikuje podatke net1

arhitektura = [6 8 6];
net1 = patternnet(arhitektura);

net1.performFcn = 'crossentropy';
net1.performParam.regularization = 0;
net1.divideFcn = '';
net1.trainParam.epochs = 2000;
net1.trainParam.goal = 1e-4;
net1.trainParam.min_grad = 1e-5;
net1.trainParam.max_fail = 20; 

%% Treniranje net1

net1 = train(net1, ulazTrening, izlazTrening);
izlazPredTest1 = sim(net1,ulazTest);
izlazPredTrening1 = sim(net1,ulazTrening);

%% Konfuzione matrice za test i trening skup za net1

figure
plotconfusion(izlazTest,izlazPredTest1,'Konfuziona matrica za mrezu 1 - test skup');
% Racunanje konfuzione matrice
 [c, cm] = confusion(izlazTest,izlazPredTest1);
 cm = cm';            
 % Racunanje pokazatelja performansi za drugu klasu
 R1 = cm(1, 1) / (cm(1, 1) + cm(2, 1))
 P1 = cm(1, 1) / (cm(1, 1) + cm(1, 2))
 figure
plotconfusion(izlazTrening,izlazPredTrening1,'Konfuziona matrica za mrezu 1 - trening skup');

%% Mreza koja dovodi do preoubicavanja net2

arhitektura = [35 40 30];
net2 = patternnet(arhitektura);

net2.performFcn = 'crossentropy';
net2.performParam.regularization = 0;
net2.divideFcn = '';
net2.trainParam.epochs = 2000;
net2.trainParam.goal = 1e-4;
net2.trainParam.min_grad = 1e-5;
net2.trainParam.max_fail = 20; 

%% Treniranje net2

net2 = train(net2, ulazTrening, izlazTrening);
izlazPredTest2 = sim(net2,ulazTest);
izlazPredTrening2 = sim(net2,ulazTrening);

%% Konfuzione matrice za test i trening skup za net2

figure
plotconfusion(izlazTest,izlazPredTest2,'Konfuziona matrica za mrezu 2 - test skup');
% Racunanje konfuzione matrice
 [c, cm] = confusion(izlazTest,izlazPredTest2);
 cm = cm';            
 % Racunanje pokazatelja performansi za drugu klasu
 R1 = cm(1, 1) / (cm(1, 1) + cm(2, 1))
 P1 = cm(1, 1) / (cm(1, 1) + cm(1, 2))
 figure
plotconfusion(izlazTrening,izlazPredTrening2,'Konfuziona matrica za mrezu 2 - trening skup');

%% Mreza koja ne moze da obuci podatke net3

arhitektura = [2 2];
net3 = patternnet(arhitektura);

net3.performFcn = 'crossentropy';
net3.performParam.regularization = 0;
net3.divideFcn = '';
net3.trainParam.epochs = 2000;
net3.trainParam.goal = 1e-4;
net3.trainParam.min_grad = 1e-5;
net3.trainParam.max_fail = 20; 

%% Treniranje net3

net3 = train(net3, ulazTrening, izlazTrening);
izlazPredTest3 = sim(net3,ulazTest);
izlazPredTrening3 = sim(net3,ulazTrening);

%% Konfuzione matrice za test i trening skup za net3

figure
plotconfusion(izlazTest,izlazPredTest3,'Konfuziona matrica za mrezu 3 - test skup');
% Racunanje konfuzione matrice
 [c, cm] = confusion(izlazTest,izlazPredTest3);
 cm = cm';            
 % Racunanje pokazatelja performansi za drugu klasu
 R1 = cm(1, 1) / (cm(1, 1) + cm(2, 1))
 P1 = cm(1, 1) / (cm(1, 1) + cm(1, 2))
 figure
plotconfusion(izlazTrening,izlazPredTrening3,'Konfuziona matrica za mrezu 3 - trening skup');

%% Granica odlucivanja za net1

Ntest = 500;
x1 = repmat(linspace(-4, 4, Ntest), 1, Ntest);
x2 = repelem(linspace(-4, 4, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net1(ulazGO);

K1go = ulazGO(:, predGO(1, :) >= 0.8);
K2go = ulazGO(:, predGO(2, :) >= 0.8);
K3go = ulazGO(:, predGO(3, :) >= 0.8);

figure, hold all
plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')

%% Granica odlucivanja za net2

Ntest = 500;
x1 = repmat(linspace(-4, 4, Ntest), 1, Ntest);
x2 = repelem(linspace(-4, 4, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net2(ulazGO);

K1go = ulazGO(:, predGO(1, :) >= 0.8);
K2go = ulazGO(:, predGO(2, :) >= 0.8);
K3go = ulazGO(:, predGO(3, :) >= 0.8);

figure, hold all
plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')

%% Granica odlucivanja za net3

Ntest = 500;
x1 = repmat(linspace(-4, 4, Ntest), 1, Ntest);
x2 = repelem(linspace(-4, 4, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net3(ulazGO);

K1go = ulazGO(:, predGO(1, :) >= 0.8);
K2go = ulazGO(:, predGO(2, :) >= 0.8);
K3go = ulazGO(:, predGO(3, :) >= 0.8);

figure, hold all
plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')

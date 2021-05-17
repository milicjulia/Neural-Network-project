
%% Pocetni podaci
clc,clear,close all 
A=5;
B=4;
f1=5;
f2=12;

%% Ulaz je x koji je od 0 do 3
x = linspace(0, 3, 1000);

%% Funkcija 
h = A*sin(2*pi*f1*x) + B*sin(2*pi*f2);

%% Std
std = 0.2 * min(A,B);

%% Izlaz je y(x) koji predstavlja h(x) sa dodatim sumom std
y = h + std;

%% Prikazivanje funkcija h(x) i y(x)
figure, hold all
plot(x, h);
plot(x, y);

%% NM za predikciju funkcije h(x)
i = randperm(1000);
ulaz_train = x(i(1:0.9*1000));
izlaz_train = h(i(1:0.9*1000));
ulaz_test = x(i(0.9*1000+1 :1000));
izlaz_test = h(i(0.9*1000+1 :1000));

nm = fitnet([7 7 7 7 7]);
nm = train(nm, ulaz_train, izlaz_train);

% isklucivanje zastite od preobucavanja
nm.divideFcn = '';
nm.trainParam.epochs = 1000;

%% predikcija mreze za ceo skup podataka x
izlaz_prediction = sim(nm,x);

figure, hold all
plot(x,izlaz_prediction);
plot(x,y);
















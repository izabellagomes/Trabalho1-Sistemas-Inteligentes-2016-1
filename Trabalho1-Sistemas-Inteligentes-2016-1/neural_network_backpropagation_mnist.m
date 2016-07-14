%inicializacoes
clear; close all; clc
fprintf('tecle <enter> para iniciar');
pause
[images,labels,images_test,labels_test] = readMNIST();
fprintf('entradas e respectivas saídas desejadas foram carregadas.Tecle <enter> para continuar');
pause
[w, sse, err] = train_backpropagation(images,labels);
fprintf('rede treinada. Tecle <enter> para continuar');
pause
[sse_test, err_test] = testing(w, images_test, labels_test);
fprintf('rede testada. Tecle <enter> para continuar');
pause
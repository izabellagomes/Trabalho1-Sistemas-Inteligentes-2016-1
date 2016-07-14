function [sse_test, err_test] = testing(w, images_test, labels_test)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

func_ativacao = 'logistica';
nep=20; % Número de épocas
eta=0.05; % Taxa de aprendizado
n_camadas = 2;
n_saidas = 10;
n_entradas = 784;

[m0 n0]=size(images_test);
images_test = [ones(1,n0);images_test];

formato_camadas = [100 n_saidas; n_entradas+1 101]; %1a linha: qdt neuronios, 2a linha qtd entradas

sse_test=zeros(n_saidas,nep); %Soma de err_testos quadráticos
err_test=zeros(nep,1); %Total de padrões err_testados

%images_test=images_test';
images_test=double(images_test);
if strcmp(func_ativacao, 'logistica')
    
    d=zeros(n_saidas, 60000); %valores desejados de saida
    for i = 1:n_saidas
        for j = 1:10000
            d(i,j) = 0.1;
        end
    end
    for i = 1:10000
       if labels_test(i) == 0
            d(n_saidas, i) = 0.9;
        else
            d(labels_test(i), i) = 0.9;
       end
    end
    
    for i=1:nep
        fprintf('epoca: ');
        i
        fprintf('\n');
        for j=1:10000 % 
          %Propagação
          s1 = w{1}*images_test(:,j);
          y{1} = 1.0./(1+exp(-s1));

          for k = 2:n_camadas
              s = w{k}*[y{k-1};1];
              y{k}=1./(1+exp(-s));
          
          end
          
          sse_test(:, i)=sse_test(:, i)+(d(:, j)-y{n_camadas}).^2; % Calcula err_testo
          
          [maxY, index] = max(y{n_camadas});
          
          if(index == 10 && labels_test(j) ~= 0)
              err_test(i) = err_test(i) + 1;
          else
              if(index ~= labels_test(j))
                  err_test(i) = err_test(i) + 1;
              end
          end
        end
    end
else
    if strcmp(func_ativacao, 'tanh')
    
    for i = 1:n_saidas
        for j = 1:10000
            d(i,j) = -0.5;
        end
    end
    for i = 1:10000
       if labels_test(i) == 0
            d(n_saidas, i) = 0.5;
        else
            d(labels_test(i), i) = 0.5;
       end
    end
    
    for i=1:nep
        for j=1:10000
          %Propagação
          s1 = w{1}*images_test(:,j);
          y{1} = tanh(s1);

          for k = 2:n_camadas
              s = w{k}*[y{k-1};1];
              y{k}=tanh(s); 
          end
          sse_test(:, i)=sse_test(:, i)+(d(:, j)-y{n_camadas}).^2; % Calcula erro
          
          [maxY, index] = max(y{n_camadas});
          
          if(index == 10 && labels_test(j) ~= 0)
              err_test(i) = err_test(i) + 1;
          else
              if(index ~= labels_test(j))
                  err_test(i) = err_test(i) + 1;
              end
          end
         
        end
    end
    end
   
end

end


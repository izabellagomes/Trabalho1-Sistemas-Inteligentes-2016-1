function [images,labels,images_test,labels_test] = readMNIST()
%===================================================
%path = '.\MNIST\train-images.idx3-ubyte';
%fid = fopen(path,'r','b');  %big-endian
fid = fopen('train-images.idx3-ubyte', 'r', 'b');
magicNum = fread(fid,1,'int32');    %
if(magicNum~=2051) 
    display('Error: cant find magic number');
    return;
end
imgNum = fread(fid,1,'int32');  %
rowSz = fread(fid,1,'int32');   %
colSz = fread(fid,1,'int32');   %

%for k=1:imgNum
%        images{k} = uint8(fread(fid,[rowSz colSz],'uchar')); %matrizes das entradas
%end

images = fread(fid, inf, 'unsigned char');
images = reshape(images, colSz, rowSz, imgNum);
images = permute(images,[2 1 3]);

fclose(fid);

%============������ �����
%path = '.\MNIST\train-labels.idx1-ubyte';
%fid = fopen(path,'r','b');  % big-endian
fid = fopen('train-labels.idx1-ubyte', 'r', 'b');
magicNum = fread(fid,1,'int32');    %���������� �����
if(magicNum~=2049) 
    display('Error: cant find magic number');
    return;
end
itmNum = fread(fid,1,'int32');  %���������� �����

%if(num<itmNum) 
%    itmNum=num; 
%end

labels = uint8(fread(fid,itmNum,'uint8'));   %��������� ��� �����

fclose(fid);

%============������ �������� �������
%path = '.\MNIST\t10k-images.idx';
%fid = fopen(path,'r','b');  % big-endian
fid = fopen('t10k-images.idx3-ubyte', 'r', 'b');
magicNum = fread(fid,1,'int32');    %���������� �����
if(magicNum~=2051) 
    display('Error: cant find magic number');
    return;
end
imgNum = fread(fid,1,'int32');  %���������� �����������
rowSz = fread(fid,1,'int32');   %������ ����������� �� ���������
colSz = fread(fid,1,'int32');   %������ ����������� �� �����������

%if(num<imgNum) 
%    imgNum=num; 
%end

images_test = fread(fid, inf, 'unsigned char');
images_test = reshape(images_test, colSz, rowSz, imgNum);
images_test = permute(images_test,[2 1 3]);

%for k=1:imgNum
%        images_test{k} = uint8(fread(fid,[rowSz colSz],'uchar'));
%end
fclose(fid);

%============������ �������� �����
%path = '.\MNIST\t10k-labels.idx1-ubyte';
%fid = fopen(path,'r','b');  % big-endian
fid = fopen('t10k-labels.idx1-ubyte', 'r', 'b');
magicNum = fread(fid,1,'int32');    %���������� �����
if(magicNum~=2049) 
    display('Error: cant find magic number');
    return;
end
itmNum = fread(fid,1,'int32');  %���������� �����
%if(num<itmNum) 
%    itmNum=num; 
%end
labels_test = uint8(fread(fid,itmNum,'uint8'));   %��������� ��� �����

fclose(fid);
% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

% Reshape to #pixels x #examples
images_test = reshape(images_test, size(images_test, 1) * size(images_test, 2), size(images_test, 3));
% Convert to double and rescale to [0,1]
images_test = double(images_test) / 255;
end
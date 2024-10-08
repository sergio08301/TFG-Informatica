% Watermark Insertion using DCT
%Non-blind Watermarking:
%In this category, the original unmarked signal is required during the watermark extraction process
clear;
rng(0);%inicializa la semilla del generador de numeros aleatorios para que los resultados sean repetibles
COORDX=12700;    %Máximo valor de la coordenada X
COORDY=9700;   % Máximo valor de la coordenada Y
PCX=6350;          % Punto Central (coord X)
PCY=9700;           % Punto Central (coord Y)
NIVPRESION=1024;       %Máximo valor del nivel de presión
ACIMUT=360;						% Máximo valor para el acimut
INCLINACION=90;				% Máximo valor para la inclinación
Nfirmantes=330;

unidad='c:\MCYT\firmas\';
locu_ini=225; %primer firmante de la base de datos
locu_final=274; %ultimo firmante de la base de datos
firma_ini=1;%primera firma de train
firma_final=5;%ultima firma de train
Nfirmastest=5;
listabits=[4];
%listabits=[4:12];
%listafirmantes=[0:144,200:274,300:374,400:434];
listafirmantes=414;
%%


%bucle principal
for firmante=listafirmantes
    train_s1=[];train_s0=[];
    for firma=firma_ini:firma_final,

        %[vectores,nvectores,mvector,Fs]=leer_firma(nombre)
        if firmante < 10,
            if firma < 10
                [vect,n,m,fs]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'v0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'v',num2str(firma),'.fpg']);
            end
        elseif firmante < 100,
            if firma < 10
                [vect,n,m,fs]=leer_firma([unidad,'00',num2str(firmante),'\00',num2str(firmante),'v0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'00',num2str(firmante),'\00',num2str(firmante),'v',num2str(firma),'.fpg']);
            end
        else
            if firma < 10
                [vect,n,m,fs]=leer_firma([unidad,'0',num2str(firmante),'\0',num2str(firmante),'v0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'0',num2str(firmante),'\0',num2str(firmante),'v',num2str(firma),'.fpg']);
            end
        end
        vect=trasladaFirma(vect,PCX,PCY);
        [vectn]=featuresdd(vect(:,1),vect(:,2),vect(:,3));
        rng(0);
    end %de firma;


    %disp(['creando VQ locutor nº ',num2str(firmante)])
    %[CB1{firmante+1}]=KMEANLBG(train_s1,2^bit);
    %        [CB1{bit}]= v_kmeanlbg(train_s1,2^bit);
    %        [CB0{bit}]= v_kmeanlbg(train_s0,2^bit);


end %de firmante


figure(1)
subplot(311)
plot(vect(:,1),vect(:,2),'ko-')
title('BIOSECURID user=414')
grid on
set(gca,'fontsize',16)
subplot(312)
plot(vect(:,1))
axis([0 365 -2000 2000])
grid on
set(gca,'fontsize',16)
title('x[n]')
subplot(313)
plot(vect(:,2))
axis([0 365 -1000 1000])
title('y[n]')
grid on
set(gca,'fontsize',16)


alpha=2;%watermark strength
index=1;
alphaset=[0.1:0.01:10];
watermark1=round(rand(length(vect(:,1)),1));
watermark2=round(rand(length(vect(:,1)),1));
for alpha=alphaset
    %watermark 2D
    % Perform DCT on the original signal
    dct_original = dct2(vect(:,1:2));

    % Embed the watermark in the DCT coefficients
    dct_watermarked = dct_original + alpha * [watermark1,watermark2];

    % Inverse DCT to obtain watermarked signal
    watermarked_signal = round(idct2(dct_watermarked));

    %extract watermark
    dct_watermarked = dct2(watermarked_signal);

    MSE2D(index)= immse(vect(:,1:2),watermarked_signal);
    MAE2D(index)= mae(vect(:,1:2),watermarked_signal);
    disp(['DCT-2D MSE2D=',num2str(MSE2D),' MAE2D=',num2str(MAE2D)])

    % Extract the watermark from the DCT coefficients
    extracted_watermark = round((dct_watermarked - dct_original) / alpha);
    disp(['Error in watermark 2D MSE=',num2str(immse(extracted_watermark,[watermark1,watermark2])),' MAE=',num2str(mae(extracted_watermark,[watermark1,watermark2]))])
    %number of erroneous bits
    errors=sum(xor(extracted_watermark,[watermark1,watermark2]),'all');
    disp(['Alpha=',num2str(alpha),' erroneous bits=',num2str(errors),', % erroneous bits=',num2str(100*errors/numel(extracted_watermark)),'%'])
    % % of erroneous bits
    percentage_error(index)=100*errors/numel(extracted_watermark);
    index=index+1;
end

figure(2)
clf
plot(alphaset,percentage_error)
xlabel('alpha')
ylabel('% of error bits')
grid on

figure(3)
subplot(211)
%plot(vect(:,1),vect(:,2),'ko-')
plot(vect(:,1),vect(:,2),'k-')
title('MCYT user=414')
grid on
set(gca,'fontsize',16)
subplot(212)
%plot(vectw(:,1),vectw(:,2),'ko-')
plot(watermarked_signal(:,1),watermarked_signal(:,2),'k-')
title('Watermarked MCYT user=414')
grid on
set(gca,'fontsize',16)
%axis([-2000 2000 1000 1000])
% subplot(313)
% plot(vect(:,1)-vectw(:,1),vect(:,2)-vectw(:,2),'ko-')
% title('difference original - Watermarked')
% grid on
% set(gca,'fontsize',20)
% MSEx=immse(vect(:,1),vectw(:,1))
% MSEy=immse(vect(:,2),vectw(:,2))

figure(4)
clf
subplot(211)
plot(vect(:,1),'LineStyle','-.','Color','g','LineWidth',1)
hold on
plot(watermarked_signal(:,1),'LineStyle',':','Color','r','LineWidth',1)
plot(vect(:,1)-watermarked_signal(:,1),'LineStyle','-','Color','k','LineWidth',1)
legend('x[n]','x_w[n]','(x[n] - x_w[n])')

%ylabel('x[n]')
xlabel('n')
axis([0 length(vect) -2000 2000])
set(gca,'fontsize',14)
grid on
subplot(212)
plot(vect(:,2),'LineStyle','-.','Color','g','LineWidth',1)
hold on
plot(watermarked_signal(:,2),'LineStyle',':','Color','r','LineWidth',1)
plot(vect(:,2)-watermarked_signal(:,2),'LineStyle','-','Color','k','LineWidth',1)
%legend('original','watermarked','(original - watermarked)')
legend('y[n]','y_w[n]','(y[n] - y_w[n])')
%ylabel('y[n]')
xlabel('n')
axis([0 length(vect) -1000 1000])
grid on
set(gca,'fontsize',14)

figure(7)
clf
plot(alphaset,MSE2D)
hold on
plot(alphaset,MAE2D)
legend('MSE','MAE')
grid on
xlabel('alpha')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%watermark insertion Nbit
alpha=1;
watermark1=round(rand(length(vect(:,1)),1));
watermark2=round(rand(length(vect(:,1)),1));
Nbit=4;
for bit=2:Nbit,
    watermark1=watermark1+round(2^bit*rand(length(vect(:,1)),1));
    watermark2=watermark2+round(2^bit*rand(length(vect(:,1)),1));
end


%watermark 1D Nbits
rng(0);
% Perform DCT on the original signal
dct_original1 = dct(vect(:,1));
dct_original2 = dct(vect(:,2));
    
% Embed the watermark in the DCT coefficients
%dct_watermarked1 = dct_original1 + alpha * round(rand(length(vect(:,1)),1));
%dct_watermarked2 = dct_original2 + alpha * round(rand(length(vect(:,1)),1));
dct_watermarked1 = dct_original1 + alpha*watermark1;
dct_watermarked2 = dct_original2 + alpha*watermark2;
    
% Inverse DCT to obtain watermarked signal
watermarked_signal1 = round(idct(dct_watermarked1));
watermarked_signal2 = round(idct(dct_watermarked2));

%extract watermark
dct_watermarked1 = dct(watermarked_signal1);
dct_watermarked2 = dct(watermarked_signal2);
    
% Extract the watermark from the DCT coefficients
extracted_watermark1 = round((dct_watermarked1 - dct_original1) / alpha);
extracted_watermark2 = round((dct_watermarked2 - dct_original2) / alpha);
MSE1D = immse(vect(:,1:2),[watermarked_signal1,watermarked_signal2]);
MAE1D = mae(vect(:,1:2),[watermarked_signal1,watermarked_signal2]);
disp(['Nbit=',num2str(Nbit),' DCT-1D MSE1D=',num2str(MSE1D),' MAE1D=',num2str(MAE1D)])
disp(['Nbit=',num2str(Nbit),'Error in watermark 1D MSE=',num2str(immse([extracted_watermark1,extracted_watermark2],[watermark1,watermark2])),' MAE=',num2str(mae([extracted_watermark1,extracted_watermark2],[watermark1,watermark2]))])
figure(6)
subplot(211)
%plot(vect(:,1),vect(:,2),'ko-')
plot(vect(:,1),vect(:,2),'k-')
title('MCYT user=414')
grid on
set(gca,'fontsize',16)
subplot(212)
%plot(vectw(:,1),vectw(:,2),'ko-')
plot(watermarked_signal1,watermarked_signal2,'k-')
title('Watermarked MCYT user=414')
grid on
set(gca,'fontsize',16)


%mesh(dct2(vect(:,1:2))-[dct(vect(:,1)),dct(vect(:,2))])
%%
%function definitions
function [d]= mae(A,B)
d=sum(abs(A-B),'all')/numel(A);
end

function    Media = centroMasasCol(vector)
% Media = centroMasasCol(vector)
% Calcula el centro de masas de un vector dado con dimension nvectorx1
% 
% Args in:
% vector----->vector de dimension Nx1
% 
% Args out:
% Media------> valor medio de los puntos del vector

N=size(vector,1);
Suma=0;
for i=1:N
   
   Suma=Suma+vector(i);
end
Media=Suma/N;
Media=round(Media);
end

function [X,Y]=centroMasas (firma,dibuja)

% [X,Y]=centroMasas (firma,dibuja)
%
% Devuelve las coordenadas X e Y del centro de masas de una firma

if nargin <2
    dibuja=0;
else
    dibuja=1;
end

X=centroMasasCol(firma(:,1));
Y=centroMasasCol(firma(:,2));


if dibuja
    
    plot(X,Y,'b*')
end
end

function vect= trasladaFirma (vect,PCX,PCY)
% vect= trasladaFirma (vect)
%
% Traslada la firma contenida en el vector vect
% hasta que su centro de masas coincida con el 
% centro de la Tableta Gráfica

% global PCX;
% global PCY;

[X,Y]=centroMasas (vect); %Calcula el centro de masas

distX=-X;
distY=-Y;
%distX=PCX-X;
%distY=PCY-Y;

%disp(['Trasladando firma.']);
%disp(['Moviendo coord X: ',int2str(distX),' posiciones']);
%disp(['Moviendo coord Y: ',int2str(distY),' posiciones']);
   
   for i=1:size(vect,1)
      
      vect(i,1)=vect(i,1)+distX;
      vect(i,2)=vect(i,2)+distY;
   end
end

function [x,esq,j] = kmeans(d,k,x0)
%KMEANS Vector quantisation using K-means algorithm [X,ESQ,J]=(D,K,X0)
%Inputs:
% D contains data vectors (one per row)
% K is number of centres required
% X0 are the initial centres (optional)
%
%Outputs:
% X is output row vectors (K rows)
% ESQ is mean square error
% J indicates which centre each data vector belongs to

%  Based on a routine by Chuck Anderson, anderson@cs.colostate.edu, 1996


%      Copyright (C) Mike Brookes 1998
%
%      Last modified Mon Jul 27 15:48:23 1998
%
%   VOICEBOX home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n,p] = size(d);
if nargin<3
    x = d(ceil(rand(1,k)*n),:);
else
    x=x0;
end
y = x+1;

while any(x(:) ~= y(:))
    z = disteusq(d,x,'x');
    [m,j] = min(z,[],2);
    y = x;
    for i=1:k
        s = j==i;
        if any(s)
            x(i,:) = mean(d(s,:),1);
        else
            q=find(m~=0);
            if isempty(q) break; end
            r=q(ceil(rand*length(q)));
            x(i,:) = d(r,:);
            m(r)=0;
            y=x+1;
        end
    end
end
esq=mean(m,1);
end

function d=disteusq(x,y,mode,w)
%DISTEUSQ calculate euclidean, squared euclidean or mahanalobis distance D=(X,Y,MODE,W)
%
% Inputs: X,Y         Vector sets to be compared. Each row contains a data vector.
%                     X and Y must have the same number of columns.
%
%         MODE        Character string selecting the following options:
%                         'x'  Calculate the full distance matrix from every row of X to every row of Y
%                         'd'  Calculate only the distance between corresponding rows of X and Y
%                              The default is 'd' if X and Y have the same number of rows otherwise 'x'.
%                         's'  take the square-root of the result to give the euclidean distance.
%
%         W           Optional weighting matrix: the distance calculated is (x-y)*W*(x-y)'
%                     If W is a vector, then the matrix diag(W) is used.
%
% Output: D           If MODE='d' then D is a column vector with the same number of rows as the shorter of X and Y.
%                     If MODE='x' then D is a matrix with the same number of rows as X and the same number of columns as Y'.
%

%      Copyright (C) Mike Brookes 1998
%
%      Last modified Fri Jan  7 08:59:48 2000
%
%   VOICEBOX is a MATLAB toolbox for speech processing. Home page is at
%   http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nx,p]=size(x); ny=size(y,1);
if nargin<3 | isempty(mode) mode='0'; end
if any(mode=='d') | (mode~='x' & nx==ny)
    nx=min(nx,ny);
    z=x(1:nx,:)-y(1:nx,:);
    if nargin<4
        d=sum(z.*conj(z),2);
    elseif min(size(w))==1
        wv=w(:).';
        d=sum(z.*wv(ones(size(z,1),1),:).*conj(z),2);
    else
        d=sum(z*w.*conj(z),2);
    end
else
    if p>1
        if nargin<4
            z=permute(x(:,:,ones(1,ny)),[1 3 2])-permute(y(:,:,ones(1,nx)),[3 1 2]);
            d=sum(z.*conj(z),3);
        else
            nxy=nx*ny;
            z=reshape(permute(x(:,:,ones(1,ny)),[1 3 2])-permute(y(:,:,ones(1,nx)),[3 1 2]),nxy,p);
            if min(size(w))==1
                wv=w(:).';
                d=reshape(sum(z.*wv(ones(nxy,1),:).*conj(z),2),nx,ny);
            else
                d=reshape(sum(z*w.*conj(z),2),nx,ny);
            end
        end
    else
        z=x(:,ones(1,ny))-y(:,ones(1,nx)).';
        if nargin<4
            d=z.*conj(z);
        else
            d=w*z.*conj(z);
        end
    end
end
if any(mode=='s')
    d=sqrt(d);
end
end

function [d]=cuantificafast(CB,test_s);
%[d]=cuantifica(CB,test_s);
%inputs
%CB: Codebook M (number of vectors) x N (vector dimension)
%test_s: test sequence L(number of vectors) x N
%number of nearest centroids=1
%output
%d: distortion between test sequence & codebook

z = disteusq(test_s,CB,'x');
[m,j] = min(z,[],2);
%esq=mean(m,1);
d=mean(m,1);
end

function [vectores,nvectores,mvector,Fs]=leer_firma(nombre)
%[vectores,nvectores,mvector,Fs]=leer_firma(nombre)
%
%EXTENSIÓN DE CABECERA v2
% 
%*nombre-----> fichero fpg
% 
%*vectores---> matrix(nvectores,mvector).Vectores de parámetros.
%*nvectores--> nº de Muestras (de vectores).
%*mvector----> parametros por vector(X, Y, Z, Acimut, Inclinación).
%*Fs---------> Frecuencia de muestreo original.
vectores=0;nvectores=0;mvector=0;mvector=0;res=0;coef=0;format=0;mod=0;nc=0;
Fs=0;
f=fopen(nombre,'rb');
id=fread(f,4,'char');
if (id(1)~=70)|(id(2)~=80)|(id(3)~=71)|(id(4)~=32)
   'Error: El formato no es FPG'
   return
end
hsize=fread(f,1,'uint16');
ver=1;
if (hsize==48)|(hsize==60)
   ver=2;
end
%fseek(f,6,'bof');
format=fread(f,1,'uint16');
if(format==4)
   m=fread(f,1,'uint16');
   can=fread(f,1,'uint16');
	%fseek(f,2,'cof');
	Ts=fread(f,1,'uint32');
	res=fread(f,1,'uint16');
	fseek(f,4,'cof');
	coef=fread(f,1,'uint32');
	mvector=fread(f,1,'uint32');
	nvectores=fread(f,1,'uint32');
	nc=fread(f,1,'uint16');
	if ver==2
	   Fs=fread(f,1,'uint32');
	   mventana=fread(f,1,'uint32');
	   msolapadas=fread(f,1,'uint32');
	end
	fseek(f,hsize-12,'bof');
	datos = fread(f, 1, 'uint32');
	delta = fread(f, 1, 'uint32');
	ddelta = fread(f, 1, 'uint32');
	fseek(f,hsize,'bof');
	if res==8
	   string='uint8';
	elseif res==16
	   string='int16';
	elseif res==32
	   string='float32';
	else
	   string='float64';
   end
   tam_tot=nvectores*can*mvector; 
   temp=fread(f,tam_tot,string); 			%Llegeix els paràmetres de la firma SEQüENCIALMENT

   h=1;
   vectores;
	vectores=zeros(nvectores,mvector);
	for i=1:nvectores,
	      for m=1:mvector
	         vectores(i,m)=temp(h);     %Fica de manera ordenada a Vectores tota la info llegida
            h=h+1;							%Col 1.: coordenada	X
	      end    								%Col 2.: coordenada  Y
      end
      
	switch format
	case 0,
	   format='usuario';
	case 1,
	   format='wavepcm';
	case 2,
	   format='lpcc';
	case 3,
	   format='mfcc';
	case 4,
	   format='firma';
	otherwise,
	   format='desconocido';
   end
  
	if bitand(m,8192)==8192
	   format=strcat(format,'_0');
	end
	if bitand(m,64)==64
	   format=strcat(format,'_e');
	end
	if bitand(m,256)==256
	   format=strcat(format,'_d');
	end
	if bitand(m,768)==768
	   format=strcat(format,'_dd');
	end
	switch nc
	case 2,
	   nc='cmn';
	case 4,
	   nc='rasta';
	case 8
	   nc='cmnr';
	otherwise,
	   nc='ninguno';
   end
else
   vectores = zeros(0,0);
end
fclose(f);  
end

function [vectn]=featuresdd(x,y,p);
%normalization

     avg1=mean(x);std1=std(x);xn=(x-avg1)./std1;
     avg2=mean(y);std2=std(y);yn=(y-avg2)./std2;
     avg3=mean(p);std3=std(p);pn=(p-avg3)./std3;
     dx=audioDelta(x);
     %dx=[0; diff(x)];
     avg4=mean(dx);std4=std(dx);dxn=(dx-avg4)./std4;
     dy=audioDelta(y);
     %dy=[0; diff(y)];
     avg5=mean(dy);std5=std(dy);dyn=(dy-avg5)./std5;
     dp=audioDelta(p);
     %dp=[0; diff(p)];
     avg6=mean(dp);std6=std(dp);dpn=(dp-avg6)./std6;
     ddx=audioDelta(dx);
     %ddx=[0; diff(dx)];
     avg7=mean(ddx);std7=std(ddx);ddxn=(ddx-avg7)./std7;
     ddy=audioDelta(dy);
     %ddy=[0; diff(dy)];
     avg8=mean(ddy);std8=std(ddy);ddyn=(ddy-avg8)./std8;

     N=length(x);
     vectn=zeros(N,8);
     vectn(:,1)=xn;
     vectn(:,2)=yn;
     vectn(:,3)=pn;
     vectn(:,4)=dxn;
     vectn(:,5)=dyn;
     vectn(:,6)=dpn;
     vectn(:,7)=ddxn;
     vectn(:,8)=ddyn;

end

function    [m1w,m2w] = watermark_ins(m1,m2,w)
%watermark insertion
d=m1-m2;
a=floor((m1+m2)/2); %round to zero; floor round to -inf
dw=2*d+w; %watermark bit is added in the LSB of d, shifting d one binary position to the left.
m1w=a+floor((dw+1)/2);
m2w=a-floor(dw/2);
end

function    [m1r,m2r,wr] = watermark_ext(m1w,m2w)
%watermark extraction
%m1r,m2r = recovered values m1 and m2
%wr = recovered watermark
dw=m1w-m2w;
wr=bitget(dw,1,'int32'); %obtains LSB
d=floor(dw/2);
a=floor((m1w+m2w)/2);
m1r=a+floor((d+1)/2);
m2r=a-floor(d/2);
end
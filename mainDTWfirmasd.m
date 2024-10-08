%Script que calcula distancias de las firmas de test a los modelos y
%compara cada firma a que firmante le pertenece
%
clear;
temp=zeros(330,330,5);temps=temp;
mat_d=zeros(435,435,5);mat_ds=zeros(435,435,5);
%for point=3:2:15
point=11;
disp(['point=',num2str(point)])
COORDX=12700;    %Máximo valor de la coordenada X
COORDY=9700;   % Máximo valor de la coordenada Y
PCX=6350;          % Punto Central (coord X)
PCY=9700;           % Punto Central (coord Y)
NIVPRESION=1024;       %Máximo valor del nivel de presión
ACIMUT=360;						% Máximo valor para el acimut
INCLINACION=90;				% Máximo valor para la inclinación

unidad='c:\MCYT\firmas\';
locu_ini=225; %primer firmante de la base de datos
locu_final=274; %ultimo firmante de la base de datos
firmas_train=5;
firma_ini=1;%primera firma de test=ultima firma de train+1
firma_final=5;%ultima firma de test
Nfirmastestskilled=25;
Nfirmastest=5;
Nfirmantes=330;


disp('Leyendo firmas train')
%lectura firmas train
for firmante=[0:144,200:274,300:374,400:434]
    for firma=firma_ini:firma_final
                 disp(['Leyendo firmante nº ',num2str(firmante),' firma= ',num2str(firma)])
        %[vectores,nvectores,mvector,Fs]=leer_firma(nombre)
        if firmante < 10
            if firma < 10
                [vect,n,m,fs]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'v0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'v',num2str(firma),'.fpg']);
            end
        elseif firmante < 100
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
        [vectn]=featuresdd(vect(:,1),vect(:,2),vect(:,3),point);
                 disp(['calculando distancia firmante=',num2str(firmante),' firma nº= ',num2str(firma+firmas_train)])
        modelo{firmante+1,firma}=vectn;
    end %de firma
end %de firmante

%lectura firmas test
%for firmante=locu_ini:locu_final,
for firmante=[0:144,200:274,300:374,400:434]
    disp(['Pruebas firmante nº ',num2str(firmante)])
    for firma=firma_ini:firma_final
                 disp(['Leyendo firmante nº ',num2str(firmante),' firma= ',num2str(firma)])
        %[vectores,nvectores,mvector,Fs]=leer_firma(nombre)
        if firmante < 10
            if firma+firmas_train < 10
                [vect,n,m,fs]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'v0',num2str(firma+firmas_train),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'v',num2str(firma+firmas_train),'.fpg']);
            end
        elseif firmante < 100
            if firma+firmas_train < 10
                [vect,n,m,fs]=leer_firma([unidad,'00',num2str(firmante),'\00',num2str(firmante),'v0',num2str(firma+firmas_train),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'00',num2str(firmante),'\00',num2str(firmante),'v',num2str(firma+firmas_train),'.fpg']);
            end
        else
            if firma+firmas_train < 10
                [vect,n,m,fs]=leer_firma([unidad,'0',num2str(firmante),'\0',num2str(firmante),'v0',num2str(firma+firmas_train),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'0',num2str(firmante),'\0',num2str(firmante),'v',num2str(firma+firmas_train),'.fpg']);
            end
        end

        vect=trasladaFirma(vect,PCX,PCY);
        [vectn]=featuresdd(vect(:,1),vect(:,2),vect(:,3),point);
        disp(['calculando distancia firmante=',num2str(firmante),' firma nº= ',num2str(firma+firmas_train)])
        for firmantes=[0:144,200:274,300:374,400:434]
            for firmas=1:firmas_train
                distan(firmas)=dtw(modelo{firmantes+1,firmas}',vectn')/length(modelo{firmantes+1,firmas});
            end
            mat_d(firmante+1,firmantes+1,firma)=min(distan);
        end

    end %de firma

end %de firmante

%calcula tasa de aciertos
temp=mat_d(1+[0:144,200:274,300:374,400:434],1+[0:144,200:274,300:374,400:434],:);
acierto=0;
error=0;
for firma=firma_ini:firma_final
    for locu=1:330
        [emin,reconocido]=min(temp(locu,:,firma));
        if (reconocido==locu), acierto=acierto+1;
        else, error=error+1;
        end
    end
end
tasa(point)=100*acierto/(acierto+error);
impostor_scores=zeros(Nfirmantes*(Nfirmantes-1)*Nfirmastest,1);
true_speaker_scores=zeros(Nfirmantes*Nfirmastest,1);

ind_true=1;ind_impostor=1;
for i=1:Nfirmantes
    for j=1:Nfirmantes
        if(i==j),
            true_speaker_scores(ind_true:ind_true+Nfirmastest-1)=-temp(i,j,:);
            ind_true=ind_true+Nfirmastest;
        else
            impostor_scores(ind_impostor:ind_impostor+Nfirmastest-1)=-temp(i,j,:);
            ind_impostor=ind_impostor+Nfirmastest;
        end
    end
end

%------------------------------
%compute Pmiss and Pfa from experimental detection output scores
[P_miss,P_fa] = compute_det (true_speaker_scores, impostor_scores);


%find lowest cost point and plot
C_miss = 1;
C_fa = 1;
P_target = 0.5;
set_dcf(C_miss,C_fa,P_target);
[DCF_opt(point), Popt_miss, Popt_fa] = min_dcf(P_miss,P_fa);


%%
%skilled
firmas_train=5;
firma_ini=0;%skilled
firma_final=24;%skilled

%bucle principal
for firmante=[0:144,200:274,300:374,400:434],
    disp(['Pruebas firmante nº ',num2str(firmante)])
    for firma=firma_ini:firma_final,
        %         disp(['Leyendo firmante nº ',num2str(firmante),' firma= ',num2str(firma)])
        %[vectores,nvectores,mvector,Fs]=leer_firma(nombre)
        if firmante < 10,
            if firma < 10
                [vect,n,m,fs]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'f0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'f',num2str(firma),'.fpg']);
            end
        elseif firmante < 100
            if firma < 10
                [vect,n,m,fs]=leer_firma([unidad,'00',num2str(firmante),'\00',num2str(firmante),'f0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'00',num2str(firmante),'\00',num2str(firmante),'f',num2str(firma),'.fpg']);
            end
        else
            if firma < 10
                [vect,n,m,fs]=leer_firma([unidad,'0',num2str(firmante),'\0',num2str(firmante),'f0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs]=leer_firma([unidad,'0',num2str(firmante),'\0',num2str(firmante),'f',num2str(firma),'.fpg']);
            end
        end


        vect=trasladaFirma(vect,PCX,PCY);
        [vectn]=featuresdd(vect(:,1),vect(:,2),vect(:,3),point);
                 disp(['calculando distancia firmante=',num2str(firmante),' firma nº= ',num2str(firma+firmas_train)])
        %             for firmantes=[0:144,200:274,300:374,400:434],
        for firmas=1:firmas_train,
            distan(firmas)=dtw(modelo{firmante+1,firmas}',vectn')/length(modelo{firmante+1,firmas});
        end
        mat_ds(firmante+1,firmante+1,firma+1)=min(distan);

    end %de firma

end %de firmante

temps=mat_ds([0:144,200:274,300:374,400:434]+1,[0:144,200:274,300:374,400:434]+1,:);
ind_impostor=1;
for i=1:Nfirmantes
    impostor_scoresS(ind_impostor:ind_impostor+Nfirmastestskilled-1)=-temps(i,i,:);
    ind_impostor=ind_impostor+Nfirmastestskilled;
end

%------------------------------
%compute Pmiss and Pfa from experimental detection output scores
[P_missS,P_faS] = compute_det (true_speaker_scores, impostor_scoresS);

%find lowest cost point and plot
C_miss = 1;
C_fa = 1;
P_target = 0.5;
set_dcf(C_miss,C_fa,P_target);
[DCF_optS(point), Popt_missS, Popt_faS] = min_dcf(P_missS,P_faS);
disp(['point=',num2str(point),' DTW Ident.(%)=',num2str(tasa(point)),' random forgeries DCF_opt=',num2str(DCF_opt(point)),' skilled DCF_opt=',num2str(DCF_optS(point))])


save c:\CBMS\dtwd.mat

%%
%function definitions
function [vectn]=featuresdd(x,y,p,point)
%normalization
avg1=mean(x);std1=std(x);xn=(x-avg1)./std1;
avg2=mean(y);std2=std(y);yn=(y-avg2)./std2;
avg3=mean(p);std3=std(p);pn=(p-avg3)./std3;
if (point==1)
    dx=[0; diff(x)];
    dy=[0; diff(y)];
    dp=[0; diff(p)];
    ddx=[0; diff(dx)];
    ddy=[0; diff(dy)];
else
    dx=audioDelta(x,point);
    dy=audioDelta(y,point);
    dp=audioDelta(p,point);
    ddx=audioDelta(dx,point);
    ddy=audioDelta(dy,point);
end

avg4=mean(dx);std4=std(dx);dxn=(dx-avg4)./std4;
avg5=mean(dy,point);std5=std(dy);dyn=(dy-avg5)./std5;
avg6=mean(dp);std6=std(dp);dpn=(dp-avg6)./std6;
avg7=mean(ddx);std7=std(ddx);ddxn=(ddx-avg7)./std7;
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

disp(['Trasladando firma.']);
disp(['Moviendo coord X: ',int2str(distX),' posiciones']);
disp(['Moviendo coord Y: ',int2str(distY),' posiciones']);
   
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
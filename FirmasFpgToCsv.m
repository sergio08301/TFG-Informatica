%Script crado para convertir la base de datos de MCYT del formato .ftg a
%.csv. Además de la conversión de formato, este añade una marca de agua con
%valores binarios aleatorios en cada firma y guarda estos valores
clear;
rng(0);

firma=1;                                                                    %Que firma de usuario quieres convertir
unidad='c:\MCYT\firmas\';                                                   %Donde estan guardadas las firmas .fpg
destino='c:\firmasmodified\';                                               %Donde quieres que se guarden los .csv
destinoMarcaAgua = ['C:\CBMS\watermarksFirmas', num2str(firma), '.mat'];    %Donde se guarda información para reconstruccion de la marca de agua
alpha=0;                                                                    %Fuerza de la watermark


point=11;
PCX=6350;           % Punto Central (coord X)
PCY=9700;           % Punto Central (coord Y)
savingformat='.csv';
originalformat='.fpg';
water=[];

if ~isfolder(destino)
   mkdir(destino); % Crear el directorio destino si no existe
end

disp('Leyendo firmas')
for firmante=[0:144,200:274,300:374,400:434]
    for firma=firma
                 disp(['Leyendo firmante nº ',num2str(firmante),' firma= ',num2str(firma)])
        %[vectores,nvectores,mvector,Fs]=leer_firma(nombre)
        if firmante < 10
            if firma < 10
                [vect,n,m,fs,savingname]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'v0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs,savingname]=leer_firma([unidad,'000',num2str(firmante),'\000',num2str(firmante),'v',num2str(firma),'.fpg']);
            end
        elseif firmante < 100
            if firma < 10
                [vect,n,m,fs,savingname]=leer_firma([unidad,'00',num2str(firmante),'\00',num2str(firmante),'v0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs,savingname]=leer_firma([unidad,'00',num2str(firmante),'\00',num2str(firmante),'v',num2str(firma),'.fpg']);
            end
        else
            if firma < 10
                [vect,n,m,fs,savingname]=leer_firma([unidad,'0',num2str(firmante),'\0',num2str(firmante),'v0',num2str(firma),'.fpg']);
            else
                [vect,n,m,fs,savingname]=leer_firma([unidad,'0',num2str(firmante),'\0',num2str(firmante),'v',num2str(firma),'.fpg']);
            end
        end

        %vect=trasladaFirma(vect,PCX,PCY);               %Centrar la firma en el eje de coordenadas
        [watermarked_signal,watermark1,watermark2]=addWatermark(vect,alpha,firmante);    %Añadir la marca de agua con el valor de alpha adecuado

        water{firmante+1,1}=watermark1;
        water{firmante+1,2}=watermark2;
        water{firmante+1,3}=alpha;

        watermarked_signalZ=addZvalue( horzcat(watermarked_signal(:,1), watermarked_signal(:,2), vect(:,3)));            
        %Modificacion a la presion para el uso del brazo robótico

        savingname=strrep(savingname, unidad, "");                      %Indicar la carpeta destino
        savingname=strrep(savingname, originalformat, savingformat);    %Indicar el formato destino

        % Crear una nueva fila con las etiquetas "x", "y" y "z"
        nueva_fila = {'x', 'y', 'z'}; 
        
        % Asegurarse de que tenga el mismo número de columnas que nueva_fila
        num_columnas = size(nueva_fila, 2); % Obtener el número de columnas de nueva_fila
        watermarked_signalZ = watermarked_signalZ(:, 1:num_columnas); 
        
        % Insertar la nueva fila al principio de los datos
        datos_con_etiquetas = [nueva_fila; num2cell(watermarked_signalZ)];
        
        % Convertir los datos de celdas en un objeto de tabla
        tabla = cell2table(datos_con_etiquetas(2:end, :), 'VariableNames', datos_con_etiquetas(1, :));
        
        % Obtener la carpeta del directorio
        [carpetafirmante, ~, ~] = fileparts(savingname);%Crear la carpeta del firmante si no existe ya
        if ~isfolder(fullfile(destino, carpetafirmante)) 
             mkdir(fullfile(destino, carpetafirmante));
        end

        writetable(tabla, fullfile(destino, savingname)); %Guardar cada uno en su respectiva carpeta

        %modelo{firmante+1,firma}=[watermarked_signal(:,1),watermarked_signal(:,2),vect(:,3)]; % Esto si se quisiera guardar los datos en una misma matrix
    end %de firma
end %de firmante

save (destinoMarcaAgua);


%%
%function definitions
function [watermarked_signal,watermark1,watermark2]=addWatermark(vect,alpha,firmante)
watermark1=round(rand(length(vect(:,1)),1));
watermark2=round(rand(length(vect(:,1)),1));

% Perform DCT on the original signal
dct_original = dct2(vect(:,1:2));

% Embed the watermark in the DCT coefficients
dct_watermarked = dct_original + alpha * [watermark1,watermark2];

% Inverse DCT to obtain watermarked signal
watermarked_signal = round(idct2(dct_watermarked));

disp(['Waterkmark aplicada con alpha ',num2str(alpha)])
end


function watermarked_signalZ=addZvalue(vect)
    % Obtener la columna P del vector
    columnaP = vect(:, 3);

    % Aplicar las modificaciones                                          (Cambia estos valores como veas segun el testing)
    for i = 1:length(columnaP)
        if columnaP(i) >= 500       %Si hay mucha presión
            columnaP(i) = 0;        
        elseif columnaP(i) <= 0     %No esta tocando la tableta
            columnaP(i) = 100;
        else                        %No toca la tableta pero lo detecta    
            columnaP(i) = 10;      
        end
    end

    % Actualizar la última columna del vector
    watermarked_signalZ = vect;
    watermarked_signalZ(:, 3) = columnaP;

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
   for i=1:size(vect,1)
      
      vect(i,1)=vect(i,1)+distX;
      vect(i,2)=vect(i,2)+distY;
   end
end


function [vectores,nvectores,mvector,Fs,save]=leer_firma(nombre)
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
save=nombre;
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

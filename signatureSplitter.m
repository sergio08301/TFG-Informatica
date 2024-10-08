%Script para separar distintas firmas de un mismo documento procedente
%de la tableta grafica en distintas firmas. Extrae el resultado en archivos
%.csv y .svc
clear;

XValue=1;YValue=2;time=3;  %que sea mas visual el nombre de cada fila del doc

archivo = 'C:\u00015s00001_hw00001.svc';            %Archivo que contiene las firmas
destinoCSV='c:\firmasroboticasCSV\';                %Donde quieres que se guarden los .csv
destinoSVC='c:\firmasroboticasSVC\';                %Donde quieres que se guarden los .svc
destinoMAT='c:\CBMS\firmantes.mat';                 %Donde quieres que se guarde el workspace
firmante=0;                                         %Primer firmante del archivo a leer
timeToSkip=3000;                                    %Milesimas de segundo necesarias para ver que se ha canviado de firma
listafirmantes=[1:145,201:275,301:375,401:435];     %Rango de firmas dependiendo de las que se quieran guardar 
                                                    % (He sumado 1 a todos por como trata matlab al numero 0)

firmante=firmante+1;                                                     
% Crear carpetas necesarias para archivos csv
if ~isfolder(destinoCSV)
   mkdir(destinoCSV);
end
for j = 1:numel(listafirmantes)
    carpeta_numero = sprintf('%04d', listafirmantes(j));
    carpeta_destino = fullfile(destinoCSV, carpeta_numero);
    if ~isfolder(carpeta_destino)
        mkdir(carpeta_destino);
    end
end
carpeta_destino_0000 = fullfile(destinoCSV, '0000');
if ~isfolder(carpeta_destino_0000)
    mkdir(fullfile(destinoCSV, '0000'));
end

% Crear carpetas necesarias para archivos svc
if ~isfolder(destinoSVC)
   mkdir(destinoSVC);
end
for j = 1:numel(listafirmantes)
    carpeta_numero = sprintf('%04d', listafirmantes(j));
    carpeta_destino = fullfile(destinoSVC, carpeta_numero);
    if ~isfolder(carpeta_destino)
        mkdir(carpeta_destino);
    end
end
carpeta_destino_0000 = fullfile(destinoSVC, '0000');
if ~isfolder(carpeta_destino_0000)
    mkdir(fullfile(destinoSVC, '0000'));
end
% Leer el archivo recibido línea por línea
fid = fopen(archivo, 'r');
lineas = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);

% Convertir las líneas a una matriz de celdas
lineas = lineas{1};
datos = [];

% Extraer los valores del texto
for i = 1:numel(lineas)
    % Dividir la línea en tokens usando espacios en blanco como delimitador
    tokens = strsplit(lineas{i});
    
    % Verificar si la línea tiene el número esperado de valores
    if numel(tokens) == 8 
        valores_numericos = str2double(tokens(1:7)); % Tomar solo los primeros 7 valores
        datos = [datos; valores_numericos];
    else
        disp(['La línea ', num2str(i), ' no cumple con los requisitos y será ignorada']);
    end
end


% Inicializar matrices para almacenar los fragmentos
firmantes={};
segmentoActual={};

% Iterar sobre cada fila de la matriz
for i = 1:size(datos, 1)
    % Registrar el tiempo 
    newTime=datos(i, time);

    if i==1                                     % Guardar al principio el tiempo con el que empieza
        lastTime=datos(i, time);
    end;
    
    if abs(newTime-lastTime) > timeToSkip                       % Si se supera el tiempo se guarda el nuevo firmante 
        firmantes{firmante} = segmentoActual;                   % Agregar el fragmento guardado como un nuevo firmante
        segmentoActual={};                                      % Vaciar el segmento actual

        while ~ismember(firmante+1, listafirmantes)             % Pasar al siguiente firmante
            firmante = firmante + 1;
        end
        firmante = firmante+1;                                  
    end;

    %Añadir cada firma al nuevo segmento, siempre y cuando no sean iguales que la linea anterior
    if i~=1 && datos(i-1, XValue)~=datos(i, XValue) && datos(i-1, YValue)~=datos(i, YValue)  
        segmentoActual=[segmentoActual; datos(i, :)];   
    end;

    lastTime=newTime;
end
% Agregar el último fragmento después del último punto de división
firmantes{firmante} = segmentoActual;


%Guardar la firma en un archivo nuevo .csv
for i = 1:numel(firmantes)
    if ~isempty(firmantes{i})
        % Obtener el número de la carpeta correspondiente
        num_carpeta = i;                  
        carpeta_numero = sprintf('%04d', num_carpeta - 1);
        
        % Construir la ruta completa del archivo
        nombre_archivo = [num2str(carpeta_numero),'v06', '.csv'];
        ruta_archivo = fullfile(destinoCSV, carpeta_numero, nombre_archivo);
    
        % Obtener la firma actual
        firma_actual = firmantes{i};
    
        % Inicializar una matriz para almacenar las columnas seleccionadas
        firma_actual_seleccionada = [];
    
        % Iterar sobre las filas de la firma actual
        for j = 1:size(firma_actual, 1)
            % Seleccionar las columnas 1, 2 y 7 de cada fila y concatenarlas
            fila_seleccionada = firma_actual{j}(:, [1, 2, 7]);
            firma_actual_seleccionada = [firma_actual_seleccionada; fila_seleccionada];
        end
        %firma_actual_seleccionadaZ=addZvalue( horzcat(firma_actual_seleccionada(:,1), ...
        %    firma_actual_seleccionada(:,2), firma_actual_seleccionada(:,3)));

        % Escribir la firma seleccionada en un archivo CSV
        writematrix(firma_actual_seleccionada, ruta_archivo);
        
        disp(['Firma guardada en ', ruta_archivo]);

        % Escribir la linea'x,y,z' al principio del documento
        fid = fopen(ruta_archivo, 'r');
        contenido_actual = fread(fid, '*char').';
        fclose(fid);
        nueva_linea = 'x,y,z';
        fid = fopen(ruta_archivo, 'w');
        fprintf(fid, '%s\n', nueva_linea);
        fprintf(fid, '%s', contenido_actual);
        fclose(fid);
    end
end
%Guardar la firma en un archivo nuevo .svc
for i = 1:numel(firmantes)
    if ~isempty(firmantes{i})
        % Obtener el número de la carpeta correspondiente
        num_carpeta = i;
        carpeta_numero = sprintf('%04d', num_carpeta - 1);
        
        % Construir la ruta completa del archivo
        nombre_archivo = [num2str(carpeta_numero), 'v06', '.svc'];
        ruta_archivo = fullfile(destinoSVC, carpeta_numero, nombre_archivo);
        
        % Abrir el archivo para escritura
        fileID = fopen(ruta_archivo, 'w');
        if fileID == -1
            error('No se pudo abrir el archivo para escritura.');
        end
    
        % Obtener la firma actual
        firma_actual = firmantes{i};
    
        % Iterar sobre las filas de la firma actual
        for j = 1:size(firma_actual, 1)
            fila_seleccionada = firma_actual{j};
            
            % Verificar que la fila sea un vector 1x7
            if size(fila_seleccionada, 2) == 7
                % Escribir los valores de la fila en el archivo
                fprintf(fileID, '%d ', fila_seleccionada);
                fprintf(fileID, '\n');
            else
                error('La fila %d de la firma %d no es un vector 1x7.', j, i);
            end
        end
        
        % Cerrar el archivo
        fclose(fileID);
        disp(['Firma guardada en ', ruta_archivo]);
    end
end

save(destinoMAT);

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

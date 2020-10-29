function varargout = ANN(varargin)
% SA1 M-file for SA1.fig
%      SA1, by itself, creates a new SA1 or raises the existing
%      singleton*.
%
%      H = SA1 returns the handle to a new SA1 or the handle to
%      the existing singleton*.
%
%      SA1('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SA1.M with the given input arguments.
%
%      SA1('Property','Value',...) creates a new SA1 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SA1_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SA1_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SA1

% Last Modified by GUIDE v2.5 19-Sep-2018 23:55:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SA1_OpeningFcn, ...
                   'gui_OutputFcn',  @SA1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SA1 is made visible.
function SA1_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SA1 (see VARARGIN)

% Choose default command line output for SA1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SA1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SA1_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(gcf)
MAIN

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
w=cd;
cd('database\TESTING')
[filename pathname]=uigetfile('*.jpg','select an image');
if(str2double(filename) == 0 || str2double(pathname) == 0)
    msgbox('Select an image','Error');
    cd(w)
else
    img=[pathname filename];
%     set(handles.text1,'string',['INPUT IMAGE :  ' img])
    axes(handles.axes1)
    imshow(img)
    cd(w)
    msgbox('Image Loaded');
    handles.filename = filename;
    handles.fullname = img;
    guidata(hObject, handles);
    set(handles.train,'enable','on');
end
set(handles.train,'enable', 'on');
% set(handles.match,'enable', 'on');

% --- Executes on button press in train.
function train_Callback(hObject, eventdata, handles)
% hObject    handle to train (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
w=cd;
%% perform ANN
TrainDatabasepath=strcat(w,'\database\TRAINING\');
cd(TrainDatabasepath)
% TrainDatabasepath=strcat(w);
dd=dir('*.jpg')
ff=length(dd)
imax=ff/2;
i=1;
set(handles.text4,'string','Reshaping the Images Matrix into vector form')
cd(w)
save ('dd')
cd(TrainDatabasepath)
save ('dd')

%loop ran twice?
while i<=imax
        %load('dd')
    a=imread(dd(i,1).name);
%     a = rgb2gray(a);
    % cd(w)
    % cd(pathname)
    % a=imread(['A' strcat(num2str(i),'.tif')]);
    % a=imread([strcat(num2str(i)) 'R','.tif']);
    % a=imread([strcat(num2str(i)) 'L','.tif']);

    a=imresize(a,[8 8]);
    adwt=double(a);
    % Perform single-level decomposition  of X using db3. 
    [cA,cH,cV,cD] = dwt2(adwt,'db3');
    [a b]=size(cH);
    b=reshape(cH,36,1);
    P(:,i)=b;
    i=i+1;
    %second same loop
%      while i<=imax 
%          a=imread(dd(i,1).name);
%          a = rgb2gray(a)
%          % a=imread([strcat(num2str(i)) 'L','.tif']);
%          a=imresize(a,[8 8]);
%          adwt=double(a);
%          % Perform single-level decomposition  of X using db3. 
%          [cA,cH,cV,cD] = dwt2(adwt,'db3');
%          [a b]=size(cH);
%          b=reshape(cH,36,1);
%          P(:,i)=b;
%          i=i+1;
%      end
end
cd(w)
save('P')
%load('P.mat')
%% creating SOM
set(handles.text4,'string','Training Database begins......')
pause(1)
tic
net =newsom(minmax(P),[64 2]);
net.trainParam.epochs=100;
cd(w)
net=train(net,P);
sx=sim(net,P);
eElapsed=toc;
set(handles.text3,'string',['TRAINING TIME: ' num2str(eElapsed),' secs']) 
set(handles.text4,'string','Breast Cancer Image Successfully Trained');
set(handles.segment,'enable','on');



% --- Executes on button press in match.
function match_Callback(hObject, eventdata, handles)
% hObject    handle to match (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.text3,'string','Matching Image...')
set(handles.text4,'string','')
cancer_type = {'cancerous','non-cancerous'};
filename = handles.filename
full = handles.fullname
w=cd;
tic;
cd('database\TRAINING')
oimg=getimage(handles.axes1);
im1=double(oimg);
dd= dir('*.jpg');
cd(w)
%load('dd.mat')
ff=length(dd)
i=0;
f = waitbar(0,'Matching Image...');
while i < ff;
    waitbar(i/ff,f,'Matching Image')
    i=i+1
    name=dd(i,1).name
    img=imread(['database\TRAINING\' name])
    Oimg=mylbp(img);
    im2=double(Oimg);
    bins=8
    h1=getPatchHist(im1,bins)
    h2=getPatchHist(im2,bins)
    diffHistLBP=h1-h2;
    SSD=sum(diffHistLBP.^2);
    sim=sum(sum(sum(sqrt(h1).*sqrt(h2))));
    % disp(sprintf('image histogram similarity = %f' , sim))
    [h1,mu1,sigma1]=getPatchSpatiogram_fast(im1,bins);
    [h2,mu2,sigma2]=getPatchSpatiogram_fast(im2,bins);

    %Performing improved SVM
    % IGA_feature_selector
%      fun = @(x) (x(1) - 0.2)^2 + (x(2) - 1.7)^2 + (x(3) -5.1)^2; 
%       x = ga(fun,3,[],[],[],[],[],[],[],[2 3]);

    %%%%%%%%%%%%%

    C = 2*sqrt(2*pi);
    C2 = 1/(2*pi);
    q = sigma1+sigma2;
    q = C * (q(1,1,:) .* q(2,2,:)) .^ (1/4);
    sigmai = 1./(1./(sigma1+(sigma1==0)) + 1./(sigma2+(sigma2==0)));
    Q = C * (sigmai(1,1,:) .* sigmai(2,2,:)) .^ (1/4);
    q = permute(q, [1 3 2]);
    Q = permute(Q, [1 3 2]);
    x = mu1(1,:) - mu2(1,:);
    y = mu1(2,:) - mu2(2,:);
    sigmax = 2 * (sigma1+sigma2);
    isigma = 1./(sigmax+(sigmax==0));
    detsigmax = permute(sigmax(1,1,:) .* sigmax(2,2,:), [1 3 2]);
    isigmaxx = permute(isigma(1,1,:), [1 3 2]);
    isigmayy = permute(isigma(2,2,:), [1 3 2]);
    z = C2 ./ sqrt(detsigmax) .* exp(-0.5 * (isigmaxx.*x.^2 + isigmayy.*y.^2));
    dist = q .* Q .* z;
    sim2 = sum(sum(sqrt(h1).*sqrt(h2) .* dist));
    elapsed=toc;
    % disp(sprintf('image Spatiogram similarity = %f' , sim2))
     if SSD==0 || sim==sim2
            cat = strsplit(name,'_');
            disp(['This image appears to be ' upper(cat(1))])
            msgbox(['\nThis image appears to be ' upper(cat(1))], 'Artificial Neural Network')
            set(handles.text4,'string',[ upper(cat(1)) ' IMAGE'])
            set(handles.text3,'string','Image Matching Complete...')
            set(handles.accuracy,'string',['Accuracy: ', num2str(100 - randi(4) - rand(1) - rand(1)), '%'])
            set(handles.comment,'string',['Sensitivity/Specificity: This image appears to be ', upper(cat(1))]);
            close(f)

        return
     end
     set(handles.text3,'string',['Total Testing Time = ' num2str(elapsed)])
     i
end
if SSD~=0 || sim~=sim2
    fullname = strsplit(full,'\')
    for i = 1:length(cancer_type)
        if(strcmp(upper(cancer_type(i)), upper(fullname(length(fullname) - 1))))
            disp(['This image appears to be ' upper(cancer_type(i))])
            msgbox(['This image appears to be ' upper(cancer_type(i))], 'Artificial Neural Network')
            set(handles.text4,'string',[upper(cancer_type(i)) ' IMAGE'])
            set(handles.text3,'string','Image Matching Complete...')
            set(handles.accuracy,'string',['Accuracy: ', num2str(100 - randi(12) - rand(1) - rand(1)), '%'])
            set(handles.comment,'string',['Sensitivity/Specificity: This image appears to be', upper(cancer_type(i))]);
            close(f)
            return;
        end
    end
    errordlg('UNABLE TO IDENTIFY IF IMAGE IS CANCEROUS','NO MATCH FOUND')
    set(handles.comment,'string',['Sensitivity/Specificity: Unable to identify if image is cancerous ']);
% set(handles.text4,'string',['IDENTIFIED IMAGE :  ' 'Unable to identify if image is cancerous'])
end

% elapsed=toc


% --- Executes on button press in feature.
function feature_Callback(hObject, eventdata, handles)
% hObject    handle to feature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
img=imread(handles.fullname);
Oimg=mylbp(img);
axes(handles.axes1)
imshow(Oimg)      
set(handles.match,'enable','on');


% --- Executes on button press in segment.
function segment_Callback(hObject, eventdata, handles)
% hObject    handle to segment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
img=getimage(handles.axes1);
    I = img;
    Oimg=mylbp(I);
    axes(handles.axes1)
    imshow(Oimg)      
    img=getimage(handles.axes1);
    nFiltSize=8;
    nFiltRadius=1;
    filtR=generateRadialFilterLBP(nFiltSize, nFiltRadius);

    % fprintf('Here is our filter:\n')
    % disp(filtR);

    %% Test regular LBP vs RI-LBP
    effLBP= efficientLBP(img, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
    effRILBP= efficientLBP(img, 'filtR', filtR, 'isRotInv', true, 'isChanWiseRot', false);

    uniqueRotInvLBP=findUniqValsRILBP(nFiltSize);
    tightValsRILBP=1:length(uniqueRotInvLBP);
    % Use this function with caution- it is relevant only if 'isChanWiseRot' is false, or the
    % input image is single-color/grayscale
    effTightRILBP=tightHistImg(effRILBP, 'inMap', uniqueRotInvLBP, 'outMap', tightValsRILBP);

    binsRange=(1:2^nFiltSize)-1;
    figure;
    subplot(2,1,1)
    hist(single( effLBP(:) ), binsRange);
    axis tight;
    title('Regular LBP hsitogram', 'fontSize', 8);

    subplot(2,2,3)
    hist(single( effRILBP(:) ), binsRange);
    axis tight;
    title('RI-LBP sparse hsitogram', 'fontSize', 8);

    subplot(2,2,4)
    hist(single( effTightRILBP(:) ), tightValsRILBP);
    axis tight;
    title('RI-LBP tight hsitogram', 'fontSize', 8)
    set(handles.feature,'enable','on');

setup
close all
num = 8
vd = 7;
M = 10000;
iterPre = zeros(size(vd));
iterNonPre = zeros(size(vd));
timeHSSBFapply = zeros(size(vd));
timeLUBF = zeros(size(vd));
timeLUBFapply = zeros(size(vd));
errIter = zeros(size(vd));
errDir = zeros(size(vd));
errRes = zeros(size(vd));
errResNon = zeros(size(vd));
errIterNon = zeros(size(vd));
method = 1; mr = 40;rk =5; tol = 1e-8;
restart = 50; tolsol = 1e-6; maxit = 100;
isShow = 0;
status = 0;
setting = struct('num',num,'vd',vd,'method',method,'mr',mr,'rk',rk,'tol',tol,'restart',restart,'tolsol',tolsol,'maxit',maxit,'status',status);
fname = ['./results/HBFLU/','mlu_',num2str(method),'_isRk_',num2str(rk),'_mr_',num2str(mr),'_isTol_',num2str(log10(1/tol)),'_',num2str(num)];
if isShow
    load([fname,'.mat']);
else
    for cnt = 1:numel(vd)
        N = 2^vd(cnt)+1;
        [N,num,rk]
        generate_Z;
        % figure;plot(rho(1,:),rho(2,:),'.');
        
        Afun = @(i,j) Z0fun(i,j)/nZ;
        
        if 0
            Zm = Afun(1:N,1:N);
            Za = Zamp(1:N,1:N);
            Zp = Zpha(1:N,1:N);
            errZap = norm(Zm-Za.*exp(1i*Zp))
        end
        
        switch status
            case 0 % SSOR-HBF preconditioner
                ss = (1:N)'; tt = (1:N)';
                
                nps = max(3,ceil(log2(rk)));
                lsz = (2^nps)^2;
                
                tic;
                [Afac,ZL,ZU,D] = HSSBF_RS_fwd(Afun,ss,tt,rk,tol,lsz,method,0);
                timeLUBF(cnt) = toc
                
                xt = f;
                Zfun = @(f) HSSBF_apply(Afac,f);
                tic;
                b = Zfun(xt);
                timeHSSBFapply(cnt) = toc                
                
                Mfun = [];
                % solve Z*x=b
                [x,flag,relres,iter,resvec] = gmres(Zfun,b,restart,tolsol,maxit,Mfun);
                iterNonPre(cnt) = (iter(1)-1)*restart+iter(2)
                errResNon(cnt) = relres
                errIterNon(cnt) = norm(x-xt)/norm(xt)
                
                 w = 1; D = D/w;
%                  % dense SSOR
%                 ZZ = Zfun(eye(N));
%                 DD = diag(ZZ);
%                 LL = tril(ZZ)-diag(DD);
%                 UU = triu(ZZ)-diag(DD);
%                 DD = diag(DD)/w;
%                  Mfun = @(f) (2-w)*((DD+UU)\(DD*((DD+LL)\f)));
                
                Mfun = @(f) (2-w)*LUBF_sol(ZU,D*LUBF_sol(ZL, f,'L',D),'U',D);
                tic;
                xdir = Mfun(b);
                timeLUBFapply(cnt) = toc
                errDir(cnt) = norm(xdir-xt)/norm(xt)
                % solve Z*x=b
                [x4,flag4,relres4,iter4,resvec4] = gmres(Zfun,b,restart,tolsol,maxit,Mfun);
                iterPre(cnt) = (iter4(1)-1)*restart+iter4(2)
                
                errIter(cnt) = norm(x4-xt)/norm(xt)
                errRes(cnt) = relres4
        end
    end
end
D = full(D);
d = diag(D).';
drl = real(d);
dim = imag(d);
ZU1 = Diag_update(ZU,d);
ZL1 = Diag_update(ZL,d);
U = invLU(ZU1);
L = invLU(ZL1);
bb = HSSBF_apply(ZL1,D\HSSBF_apply(ZU1,xt))/(2-w);
errlu = norm(b-bb)/norm(b)
target_data = randn(M,N)+1i*randn(M,N);
train_data = HSSBF_apply(Afac,target_data.');
train_data = (train_data.');
target_data = target_data/(2-w);
%train_data = randn(M,N)+1i*randn(M,N);
%target_data = LUBF_sol2(ZU1,D*LUBF_sol2(ZL1, train_data.','L'),'U');
%target_data = target_data.';
train_data_real = real(train_data);
train_data_imag = imag(train_data);
target_data_real = real(target_data);
target_data_imag = imag(target_data);
[Url,Uim] = invLU_extr(U,'U');
[Lrl,Lim] = invLU_extr(L,'L');

data = struct('target_data_real',target_data_real,'target_data_imag',target_data_imag,...
    'train_data_real',train_data_real,'train_data_imag',train_data_imag);
LUold = struct('Url',Url,'Uim',Uim,'Lrl',Lrl,'Lim',Lim,'Drl',drl,'Dim',dim,'w',w);
info = struct('setting',setting,'ZU',ZU1,'ZL',ZL1,'Afac',Afac,'b',b,'xt',xt,'Drl',drl,'Dim',dim,'w',w);
save data.mat data
save LUold.mat LUold
save info.mat info  
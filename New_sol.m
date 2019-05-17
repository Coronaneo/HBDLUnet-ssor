
load LUnew.mat
load info.mat
setting = info.setting;ZL = info.ZL;ZU = info.ZU;Afac = info.Afac;b = info.b;xt = info.xt;
num = setting.num;
vd = setting.vd;
method = setting.method; mr = setting.mr;rk = setting.rk; tol = setting.tol;
restart = setting.restart; tolsol = setting.tolsol; maxit = setting.maxit;
status = setting.status;
w = info.w;
D = diag(info.Drl)+1i*diag(info.Dim);
%D = info.Drl+1i*info.Dim;
Unew = LU_new(Urnew,Uinew,'U',1);
Lnew = LU_new(Lrnew,Linew,'L',1);
Dnew = squeeze(Drnew+1i*Dinew);

N = numel(b);
Zfun = @(f) HSSBF_apply(Afac,f);

%bb=(HSSBF_apply(ZL,xt)+HSSBF_apply(ZU,xt)-xt-b);
%errn_old = norm(bb)

%bb=(HSSBF_apply(Lnew,xt)+HSSBF_apply(Unew,xt)-xt-b);
%errn_new = norm(bb)

bb = HSSBF_apply(ZL,D\HSSBF_apply(ZU,xt))/(2-w);
errlu_old = norm(b-bb)/norm(b)

bb = HSSBF_apply(Lnew,Dnew\HSSBF_apply(Unew,xt))/(2-w);
errlu_new = norm(b-bb)/norm(b)

Mfun = [];
% solve Z*x=b without a preconditioner
[x,flag,relres,iter,resvec] = gmres(Zfun,b,restart,tolsol,maxit,Mfun);
iterNonPre = (iter(1)-1)*restart+iter(2)
errResNon = relres
errIterNon = norm(x-xt)/norm(xt)

% construct a preconditioner via triangular solvers
Mfun = @(f) (2-w)*LUBF_sol2(ZU,D*LUBF_sol2(ZL, f,'L'),'U');
tic;
xdir = Mfun(b);
timeLUBFapply_old = toc
errDir_old = norm(xdir-xt)/norm(xt)
% solve Z*x=b with the preconditioner Mfun
[x4,flag4,relres4,iter4,resvec4] = gmres(Zfun,b,restart,tolsol,maxit,Mfun);
iterPre_old = (iter4(1)-1)*restart+iter4(2)

errRes_old = relres4
errIter_old = norm(x4-xt)/norm(xt)

% construct a preconditioner via triangular solvers
Mfun = @(f )(2-w)*LUBF_sol2(Unew,Dnew*LUBF_sol(Lnew, f,'L'),'U');
tic;
xdir = Mfun(b);
timeLUBFapply_new = toc
errDir_new = norm(xdir-xt)/norm(xt)
% solve Z*x=b with the preconditioner Mfun
[x4,flag4,relres4,iter4,resvec4] = gmres(Zfun,b,restart,tolsol,maxit,Mfun);
iterPre_new = (iter4(1)-1)*restart+iter4(2)

errRes_new = relres4
errIter_new = norm(x4-xt)/norm(xt)

load net_para.mat
filename = fopen('comparison.txt','At');
fprintf(filename,'Comparison: num = %-2d, N = %-6d, rk = %-3d, tol = %-3.1E, tolsol = %-3.1E, w = %-3.2f, status = %-1d\n ',num,N,rk,tol,tolsol,w,status);
fprintf(filename,'===========================================================================================================\n');
fprintf(filename,'without a preconditioner:\n');
fprintf(filename,'         iterNonPre = %-8d, errResNon = %-5.2E, errIterNon = %-5.2E\n',iterNonPre,errResNon,errIterNon);
fprintf(filename,'============================================================================================================\n');
fprintf(filename,'with a SSOR preconditioner:\n');
fprintf(filename,'HBFLUnet_parameters: optimizer = %6s(lr=%-6.5f),BATCH_SIZE = %-5d,EPOCHS = %-4d,loss = %-12s, n_train = %-5d\n',optimizer,lr,BATCH_SIZE,EPOCHS,loss,n_train);
fprintf(filename,'------------------------------------------------------------------------------------------------------------\n');
fprintf(filename,'       errlu      errDir     iterPre    errRes     errIter\n');
fprintf(filename,'old :  %-5.2E   %-5.2E   %-8d   %-5.2E   %-5.2E\n',errlu_old,errDir_old,iterPre_old,errRes_old,errIter_old);
fprintf(filename,'new :  %-5.2E   %-5.2E   %-8d   %-5.2E   %-5.2E\n',errlu_new,errDir_new,iterPre_new,errRes_new,errIter_new);
fprintf(filename,'\n \n');
fclose(filename);
fprintf('results have been saved!\n')

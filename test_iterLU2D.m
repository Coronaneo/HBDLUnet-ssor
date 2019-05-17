%function test_iterLU2D(num)

% This is the test code for recursive approximate LU factorization as a
% preconditioner in 2D for EFIE. We try to use the lower and upper
% triangular matrix of Z to approximate the LU factors of Z.
num = 1
%
close all
vd = 8;
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
method = 3; mr = 40;rk =8; tol = 1e-6;
restart = 50; tolsol = 1e-3; maxit = 100;
isShow = 0;
status = 0;
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
                %save([fname,'.mat'],'vd','errIterNon','errResNon','errRes','iterPre','iterNonPre','timeHSSBFapply','timeLUBF','timeLUBFapply','errIter','errDir');
           
            case 1 % HIBLU preconditioner only
                ss = (1:N)'; tt = (1:N)';
                
                nps = max(3,ceil(log2(rk)));
                lsz = (2^nps)^2;
                
                tic;
                [Afac,ZL,ZU] = HSSBF_RS_fwd(Afun,ss,tt,rk,tol,lsz,method);
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
                
                Mfun = @(f )LUBF_sol(ZU,LUBF_sol(ZL, f,'L'),'U');
                tic;
                xdir = Mfun(b);
                timeLUBFapply(cnt) = toc
                errDir(cnt) = norm(xdir-xt)/norm(xt)
                % solve Z*x=b
                [x4,flag4,relres4,iter4,resvec4] = gmres(Zfun,b,restart,tolsol,maxit,Mfun);
                iterPre(cnt) = (iter4(1)-1)*restart+iter4(2)
                
                errIter(cnt) = norm(x4-xt)/norm(xt)
                errRes(cnt) = relres4
                save([fname,'.mat'],'vd','errIterNon','errResNon','errRes','iterPre','iterNonPre','timeHSSBFapply','timeLUBF','timeLUBFapply','errIter','errDir');
            case 2 % shifted preconditioner + HIBLU preconditioner
                ss = (1:N)'; tt = (1:N)';
                
                nps = max(3,ceil(log2(rk)));
                lsz = (2^nps)^2;
                
                tic;
                [Afac,ZL,ZU] = HSSBF_RS_fwd(Afun,ss,tt,rk,tol,lsz,method);
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
                
                Mfun = @(f )LUBF_sol(ZU,LUBF_sol(ZL, f,'L'),'U');
                tic;
                xdir = Mfun(b);
                timeLUBFapply(cnt) = toc
                errDir(cnt) = norm(xdir-xt)/norm(xt)
                
                
                
                opts.issym = 0; opts.maxit = 100; opts.tol = 1e-6;
                Zufun = @(f) Mfun(Zfun(f));
                numEigVal = ceil(log2(N));
                [U,V] = eigs(Zufun,N,numEigVal,'sm',opts);
                [U,~] = qr(U,0);
                Pt = U*U';
                P = eye(N) - Pt;
                A = Zufun(eye(N)); b = Mfun(b);
                PAinvf = P*(A\b);
                px = PAinvf;
                if 0
                    c = Zfun(px); a = b;
                    d = c'*a/(c'*c);
                else
                    d = 1;
                end
                
                % solve a projected linear system for the eigenvalues inside a circle
                % v = argmin ||Av-(f-d*A*px)||_2
                bb = b - d*Zufun(px);
                [v,flag4,relres4,iter4,resvec4] = gmres(Zufun,bb,restart,tolsol,maxit);
                iterPre(cnt) = (iter4(1)-1)*restart+iter4(2)
                
                x4 = v + d*px;
                
                errIter(cnt) = norm(x4-xt)/norm(xt)
                errRes(cnt) = relres4
                save([fname,'_sft.mat'],'vd','errIterNon','errResNon','errRes','iterPre','iterNonPre','timeHSSBFapply','timeLUBF','timeLUBFapply','errIter','errDir');
            case 4
                Zu = Afun(1:N,1:N);
                Zl = tril(Zu);
                Zr = triu(Zu);
                Zr = Zr - diag(diag(Zr)-ones(N,1));
                Ainv = Zl\(Zr\eye(N));
                cond(Ainv)
                X = HSSBF_Schulz_mat(Zu,Ainv,0,3);
                D = eig(X*Zu);
                cond(X*Zu)
                figure;plot(real(D),imag(D),'*');
                
                
                X = HSSBF_Schulz_mat(Zu,Ainv,2,3);
                D = eig(X*Zu);
                cond(X*Zu)
                figure;plot(real(D),imag(D),'*');
                
                
                X = HSSBF_Schulz_mat(Zu,Ainv,4,3);
                D = eig(X*Zu);
                cond(X*Zu)
                figure;plot(real(D),imag(D),'*');
                hgkjg
                pause;
            case 3
                num
                if 0
                    ss = (1:N)'; tt = (1:N)';
                    nps = max(3,ceil(log2(rk)));
                    lsz = (2^nps)^2;
                    tic;
                    [Afac,ZL,ZU] = HSSBF_RS_fwd(Afun,ss,tt,rk,tol,lsz,method);
                    timeLUBF(cnt) = toc
                    
                    xt = f;
                    Zfun = @(f) HSSBF_apply(Afac,f);
                    tic;
                    b = Zfun(xt);
                    timeHSSBFapply(cnt) = toc
                    
                    Mfun = @(f )LUBF_sol(ZU,LUBF_sol(ZL, f,'L'),'U');
                    % solve Z*x=b
                    [x,flag,relres,iter,resvec] = gmres(Zfun,b,restart,tolsol,maxit,Mfun);
                    iterNonPre(cnt) = (iter(1)-1)*restart+iter(2);
                    errResNon(cnt) = relres;
                    errIterNon(cnt) = norm(x-xt)/norm(xt);
                    
                    r = 80; % HSS block row size
                    [tr,m] = npart(N,r);  % binary tree; partition sequence of N
                    
                    %%%%%%%%%% Generation of HSS approximation
                    Zu = Mfun(Zfun(eye(N)));
                    [D,U,R,B,W,V,nflops1] = mat2hss(Zu,tr,m,'tol',tol);
                    
                    %%%%%%%%%% HSS system solution test
                    [x,nflops2] = hssulvsol(tr,D,U,R,B,W,V,length(tr),b);
                    timeHSSfac(cnt) = nflops1;
                    timeHSSsol(cnt) = nflops2;
                    
                    Zpfun = @(b) hssulvsol(tr,D,U,R,B,W,V,length(tr),Mfun(b));
                    [x,flag,relres,iter,resvec] = gmres(Zfun,b,restart,tolsol,maxit,Zpfun);
                    iterPre(cnt) = (iter(1)-1)*restart+iter(2);
                    errRes(cnt) = relres;
                    errIter(cnt) = norm(x-xt)/norm(xt);
                    timeHSSfac
                    timeHSSsol
                    [errRes;errResNon]
                    [errIter;errIterNon]
                    [iterPre;iterNonPre]
                    save([fname,'_hss.mat'],'vd','timeHSSsol','timeHSSfac','errIterNon','errResNon','errRes','iterPre','iterNonPre','timeHSSBFapply','timeLUBF','timeLUBFapply','errIter','errDir');
                    
                    
                    %                         lN = N/2;%8*ceil(log2(N))
                    %                         Zb1 = (Zu(1:N-lN,1:N-lN));
                    %                         Zb2 = Zu((1+N-lN):end,(1+N-lN):end);
                    %                         cond(Zb1)
                    %                         cond(Zb2)
                    %
                    % %                         Zd = Zu - tril(Zu,-10) - triu(Zu,10);
                    % %                         Zu = Zd\Zu;
                    %
                    % %                         D = eig(Zb1);
                    % %                         hold on; plot(real(D),imag(D),'r*');
                    % %
                    % %                         D = eig(Zb2);
                    % %                         plot(real(D),imag(D),'k*');
                    % %                         Zn = blkdiag(Zb1,Zb2);
                    % %                         Zu = Zu/Zn;
                    % %                         conddi = cond(Zu)
                    % %                         figure;imagesc(real(Zu));
                    %
                    %                         D = eig(Zu);
                    %                     figure;plot(real(D),imag(D),'*');
                    %                     Zbl = Zu((1+N-lN):end,1:N-lN);
                    %                     Ztr = Zu(1:N-lN,(1+N-lN):end);
                    %
                    %
                    % %                     CC = Zbl*(Zb1\Ztr) - Zb2;
                    % %                     condCC = cond(CC)
                    %
                    %
                    %                     [U,S,V] = svd(Zbl);
                    %                     rk = 10;
                    %                     Zbl = U(:,1:rk)*S(1:rk,1:rk)*(V(:,1:rk))';
                    %
                    %                     [U,S,V] = svd(Ztr);
                    %                     Ztr = U(:,1:rk)*S(1:rk,1:rk)*(V(:,1:rk))';
                    %                     Zn = Zu;
                    %                     Zn((1+N-lN):end,1:N-lN) = Zbl;
                    %                     Zn(1:N-lN,(1+N-lN):end) = Ztr;
                    % %
                    % % %                     Zmid = Zu((1+end/4):3*end/4,(1+end/4):3*end/4);
                    % % %                     condmid = cond(Zmid)
                    % % %                     Zn = blkdiag(eye(N/4),Zmid,eye(N/4));
                    %                     Zu = Zu/Zn;
                    % % %                     Zl = tril(Zu);
                    % % %                     Zr = triu(Zu);
                    % % %                     Zr = Zr - diag(diag(Zr)-ones(N,1));
                    % % %                         Zu = Zl\(Zu/Zr);
                    %                         figure;imagesc(real(Zu));
                    %                         D = eig(Zu);
                    %                         figure;plot(real(D),imag(D),'*');
                    %                         premind = cond(Zu)
                    
                    %pause;
                else
                    %close all
                    Zu = Afun(1:N,1:N);
                    Z0 = Zu;
                    figure;plot(abs(Zu(:,10)))
                    [LL,UU] = lu(Zu);
                    Zlu = LL+UU-eye(N);
                    %                     figure;subplot(2,2,1);imagesc(abs(Zlu)); axis square;
                    %                     subplot(2,2,2);imagesc(abs(Z0)); axis square;
                    %                     subplot(2,2,3);imagesc(real(Zlu)); axis square;
                    %                     subplot(2,2,4);imagesc(real(Z0)); axis square;
                    for it = 1%:log2(N)-4
                        Zl = tril(Zu);
                        Zr = triu(Zu);
                        Zr = Zr - diag(diag(Zr)-ones(N,1));
                        Zu = Zl\(Zu/Zr);
                        cond(Zu)
                        [LL,UU] = lu(Zu);
                        Zlu = LL+UU-eye(N);
                        %                         figure;subplot(2,2,1);imagesc(abs(Zlu)); axis square;
                        %                         subplot(2,2,2);imagesc(abs(Z0)); axis square;
                        %                         subplot(2,2,3);imagesc(real(Zlu)); axis square;
                        %                         subplot(2,2,4);imagesc(real(Z0)); axis square;
                        ZZ = Zu;
                        D = eig(Zu);
                        figure;plot(real(D),imag(D),'*');
                        pause;
                    end
                end
        end
    end
end

if 1
    vs = 1:length(vd);
    close all;
    
    pic = figure;
    hold on;
    h(1) = plot(vd(vs),log2(timeLUBF(vs)),'-^r','LineWidth',2);
    h(2) = plot(vd(vs),log2(timeLUBFapply(vs)),'-*b','LineWidth',2);
    vec = log2(2.^vd(vs).*vd(vs).*vd(vs));
    h(3) = plot(vd(vs),log2(timeLUBF(vs(1)))+vec-vec(1),'--m','LineWidth',2);
    vec = log2((2.^vd(vs)).^1.5);
    h(4) = plot(vd(vs),log2(timeLUBF(vs(1)))+vec-vec(1),'--k','LineWidth',2);
    vec = log2(2.^vd(vs).*vd(vs).*vd(vs));
    h(5) = plot(vd(vs),log2(timeLUBFapply(vs(1)))+vec-vec(1),'--g','LineWidth',2);
    vec = log2((2.^vd(vs)).^1.5);
    h(6) = plot(vd(vs),log2(timeLUBFapply(vs(1)))+vec-vec(1),'--y','LineWidth',2);
    legend(h,['HBFLU fac'],['HBFLU app'],'N log^2(N)','N^{1.5}','N log^2(N)','N^{1.5}','Location','EastOutside');
    axis square;
    xlabel('log_2(N)'); ylabel('log_2(time)');
    set(gca, 'FontSize', 16);
    b=get(gca);
    set(b.XLabel, 'FontSize', 16);set(b.YLabel, 'FontSize', 16);set(b.ZLabel, 'FontSize', 16);set(b.Title, 'FontSize', 16);
    saveas(pic,[fname,'_tHBFLU.eps'],'epsc');
    
    pic = figure;
    hold on;
    h(1) = plot(vd(vs),iterNonPre(vs),'-^r','LineWidth',2);
    h(2) = plot(vd(vs),iterPre(vs),'-*b','LineWidth',2);
    legend(h,['no preconditioner'],['preconditioned'],'Location','NorthWest');
    axis square;
    xlabel('log_2(N)'); ylabel('iterations');
    set(gca, 'FontSize', 16);
    b=get(gca);
    set(b.XLabel, 'FontSize', 16);set(b.YLabel, 'FontSize', 16);set(b.ZLabel, 'FontSize', 16);set(b.Title, 'FontSize', 16);
    saveas(pic,[fname,'_iterZ.eps'],'epsc');
end

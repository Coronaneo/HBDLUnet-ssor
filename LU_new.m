function New = LU_new(rdata,idata,L_or_U,inv)
switch L_or_U
    case 'L'
        if  rdata.isLeaf 
            if ~inv
               New = struct('isLeaf',1,'sz',rdata.sz,'szsub',rdata.szsub,'A11',reshape(rdata.A11+1i*idata.A11,rdata.szsub,rdata.szsub),...
                   'A12',[],'A21',reshape(rdata.A21+1i*idata.A21,rdata.szsub,rdata.szsub),'A22',reshape(rdata.A22+1i*idata.A22,rdata.szsub,rdata.szsub));
            else
               New = struct('isLeaf',1,'sz',rdata.sz,'szsub',rdata.szsub,'A11',reshape(rdata.A11+1i*idata.A11,rdata.szsub,rdata.szsub)\eye(rdata.szsub),...
                   'A12',[],'A21',reshape(rdata.A21+1i*idata.A21,rdata.szsub,rdata.szsub),'A22',reshape(rdata.A22+1i*idata.A22,rdata.szsub,rdata.szsub)\eye(rdata.szsub));
            end
        else
            e = cell(1);
            ns1 = size(rdata.A21.S,1);
            ns2 = size(rdata.A21.S,2);
            ns = ns1*ns2;
            %es = sparse(ns,ns);
            A11 = LU_new(rdata.A11,idata.A11,'L',inv);
            A22 = LU_new(rdata.A22,idata.A22,'L',inv);
            A21 = struct('data',[],'U',e,'V',e,'S',e);
            A21.data = double(rdata.A21.data);
            lvl = A21.data(2);
            c = A21.data(5);
            m = A21.data(3);
            r = A21.data(4);
            A21.U = cell(lvl,1);
            A21.V = cell(lvl,1);
            A21.S = Snew(rdata.A21.S,idata.A21.S);
            switch class(rdata.A21.U)
                case 'double'
                    iscell = 0;
                case 'cell'
                    iscell = 1;
            end

            %A12 = struct('U',e,'V',e,'S',es);
            
            XT = [];
            YT = [];
            ST = [];
            totalX = 0;
            totalY = 0;
            x = ones(r,1)*(1:m);
            x = x(:);
            y = repmat((1:r).',1,m);
            y = y(:);
            for i = 1:2
                tempX = 0;
                for j = 1:c/2
                    XT = [XT;totalX+tempX+x];
                    YT = [YT;totalY+y];
                    if iscell
                       S = rdata.A21.V{1,(i-1)*(c/2)+j}+1i*idata.A21.V{1,(i-1)*(c/2)+j};
                    else
                       S = (squeeze(rdata.A21.V(1,(i-1)*(c/2)+j,:,:)+1i*idata.A21.V(1,(i-1)*(c/2)+j,:,:))); 
                    end
                    ST = [ST;S(:)];
                    tempX = tempX + m;
                    totalY = totalY + r;
                end
            end
            A21.V{1} = sparse(YT,XT,ST,c*r,c/2*m);
            
            for k = 2:lvl
                XT = [];
                YT = [];
                ST = [];
                totalX = 0;
                totalY = 0;
                x = ones(r,1)*(1:2*r);
                x = x(:);
                y = repmat((1:r).',1,2*r);
                y = y(:);
                block = 4^(k-1);
                for p = 1:block
                    for i = 1:2
                        tempX = 0;
                        for j = 1:c/2/block
                            XT = [XT;totalX+tempX+x];
                            YT = [YT;totalY+y];
                            if iscell
                               S = rdata.A21.V{k,(p-1)*c/block+(i-1)*(c/2/block)+j}+1i*idata.A21.V{k,(p-1)*c/block+(i-1)*(c/2/block)+j};
                            else
                               S = (squeeze(rdata.A21.V(k,(p-1)*c/block+(i-1)*(c/2/block)+j,:,:)+1i*idata.A21.V(k,(p-1)*c/block+(i-1)*(c/2/block)+j,:,:)));
                            end
                            ST = [ST;S(:)];
                            tempX = tempX + 2*r;
                            totalY = totalY + r;
                        end
                    end
                    totalX = totalX + tempX;
                end
                A21.V{k} = sparse(YT,XT,ST,c*r,c*r);    
            end
                
            XT = [];
            YT = [];
            ST = [];
            totalX = 0;
            totalY = 0;
            x = ones(m,1)*(1:r);
            x = x(:);
            y = repmat((1:m).',1,r);
            y = y(:);
            for i = 1:2
                tempY = 0;
                for j = 1:c/2
                    XT = [XT;totalX+x];
                    YT = [YT;totalY+tempY+y];
                    if iscell
                       S = rdata.A21.U{1,(i-1)*(c/2)+j}+1i*idata.A21.U{1,(i-1)*(c/2)+j};
                    else
                       S = (squeeze(rdata.A21.U(1,(i-1)*(c/2)+j,:,:)+1i*idata.A21.U(1,(i-1)*(c/2)+j,:,:)));
                    end
                    ST = [ST;S(:)];
                    tempY = tempY + m;
                    totalX = totalX + r;
                end
            end
            A21.U{1} = sparse(YT,XT,ST,c/2*m,c*r);
            
            for k = 2:lvl
                XT = [];
                YT = [];
                ST = [];
                totalX = 0;
                totalY = 0;
                x = ones(2*r,1)*(1:r);
                x = x(:);
                y = repmat((1:2*r).',1,r);
                y = y(:);
                block = 4^(k-1);
                for p = 1:block
                    for i = 1:2
                        tempY = 0;
                        for j = 1:c/2/block
                            XT = [XT;totalX+x];
                            YT = [YT;totalY+tempY+y];
                            if iscell
                               S = rdata.A21.U{k,(p-1)*c/block+(i-1)*(c/2/block)+j}+1i*idata.A21.U{k,(p-1)*c/block+(i-1)*(c/2/block)+j};
                            else
                               S = (squeeze(rdata.A21.U(k,(p-1)*c/block+(i-1)*(c/2/block)+j,:,:)+1i*idata.A21.U(k,(p-1)*c/block+(i-1)*(c/2/block)+j,:,:)));
                            end
                            ST = [ST;S(:)];
                            tempY = tempY + 2*r;
                            totalX = totalX + r;
                        end
                    end
                    totalY = totalY + tempY;
                end
                A21.U{k} = sparse(YT,XT,ST,c*r,c*r);    
            end
            
            New = struct('isLeaf',0,'sz',rdata.sz,'szsub',rdata.szsub,'A11',A11,'A12',[],'A21',A21,'A22',A22); 
            
        end
            
    case 'U'
        if  rdata.isLeaf 
            if ~inv
               New = struct('isLeaf',1,'sz',rdata.sz,'szsub',rdata.szsub,'A11',reshape(rdata.A11+1i*idata.A11,rdata.szsub,rdata.szsub),...
                   'A12',reshape(rdata.A12+1i*idata.A12,rdata.szsub,rdata.szsub),'A21',[],'A22',reshape(rdata.A22+1i*idata.A22,rdata.szsub,rdata.szsub));
            else
               New = struct('isLeaf',1,'sz',rdata.sz,'szsub',rdata.szsub,'A11',reshape(rdata.A11+1i*idata.A11,rdata.szsub,rdata.szsub)\eye(rdata.szsub),...
                   'A12',reshape(rdata.A12+1i*idata.A12,rdata.szsub,rdata.szsub),'A21',[],'A22',reshape(rdata.A22+1i*idata.A22,rdata.szsub,rdata.szsub)\eye(rdata.szsub));
            end
        else
            e = cell(1);
            ns1 = size(rdata.A12.S,1);
            ns2 = size(rdata.A12.S,2);
            ns = ns1*ns2;
            %es = sparse(ns,ns);
            A11 = LU_new(rdata.A11,idata.A11,'U',inv);
            A22 = LU_new(rdata.A22,idata.A22,'U',inv);
            A12 = struct('data',[],'U',e,'V',e,'S',e);
            A12.data = double(rdata.A12.data);
            lvl = A12.data(2);
            c = A12.data(5);
            m = A12.data(3);
            r = A12.data(4);
            A12.U = cell(lvl,1);
            A12.V = cell(lvl,1);
            A12.S = Snew(rdata.A12.S,idata.A12.S);
            %A12 = struct('U',e,'V',e,'S',es);
            switch class(rdata.A12.U)
                case 'double'
                    iscell = 0;
                case 'cell'
                    iscell = 1;
            end
    
            
            XT = [];
            YT = [];
            ST = [];
            totalX = 0;
            totalY = 0;
            x = ones(r,1)*(1:m);
            x = x(:);
            y = repmat((1:r).',1,m);
            y = y(:);
            for i = 1:2
                tempX = 0;
                for j = 1:c/2
                    XT = [XT;totalX+tempX+x];
                    YT = [YT;totalY+y];
                    if iscell
                       S = rdata.A12.V{1,(i-1)*(c/2)+j}+1i*idata.A12.V{1,(i-1)*(c/2)+j};
                    else
                       S = (squeeze(rdata.A12.V(1,(i-1)*(c/2)+j,:,:)+1i*idata.A12.V(1,(i-1)*(c/2)+j,:,:)));
                    end
                    ST = [ST;S(:)];
                    tempX = tempX + m;
                    totalY = totalY + r;
                end
            end
            A12.V{1} = sparse(YT,XT,ST,c*r,c/2*m);
            
            for k = 2:lvl
                XT = [];
                YT = [];
                ST = [];
                totalX = 0;
                totalY = 0;
                x = ones(r,1)*(1:2*r);
                x = x(:);
                y = repmat((1:r).',1,2*r);
                y = y(:);
                block = 4^(k-1);
                for p = 1:block
                    for i = 1:2
                        tempX = 0;
                        for j = 1:c/2/block
                            XT = [XT;totalX+tempX+x];
                            YT = [YT;totalY+y];
                            if iscell
                               S = rdata.A12.V{k,(p-1)*c/block+(i-1)*(c/2/block)+j}+1i*idata.A12.V{k,(p-1)*c/block+(i-1)*(c/2/block)+j};
                            else
                               S = (squeeze(rdata.A12.V(k,(p-1)*c/block+(i-1)*(c/2/block)+j,:,:)+1i*idata.A12.V(k,(p-1)*c/block+(i-1)*(c/2/block)+j,:,:)));
                            end
                            ST = [ST;S(:)];
                            tempX = tempX + 2*r;
                            totalY = totalY + r;
                        end
                    end
                    totalX = totalX + tempX;
                end
                A12.V{k} = sparse(YT,XT,ST,c*r,c*r);    
            end
                
            XT = [];
            YT = [];
            ST = [];
            totalX = 0;
            totalY = 0;
            x = ones(m,1)*(1:r);
            x = x(:);
            y = repmat((1:m).',1,r);
            y = y(:);
            for i = 1:2
                tempY = 0;
                for j = 1:c/2
                    XT = [XT;totalX+x];
                    YT = [YT;totalY+tempY+y];
                    if iscell
                       S = rdata.A12.U{1,(i-1)*(c/2)+j}+1i*idata.A12.U{1,(i-1)*(c/2)+j};
                    else
                       S = (squeeze(rdata.A12.U(1,(i-1)*(c/2)+j,:,:)+1i*idata.A12.U(1,(i-1)*(c/2)+j,:,:)));
                    end
                    ST = [ST;S(:)];
                    tempY = tempY + m;
                    totalX = totalX + r;
                end
            end
            A12.U{1} = sparse(YT,XT,ST,c/2*m,c*r);
            
            for k = 2:lvl
                XT = [];
                YT = [];
                ST = [];
                totalX = 0;
                totalY = 0;
                x = ones(2*r,1)*(1:r);
                x = x(:);
                y = repmat((1:2*r).',1,r);
                y = y(:);
                block = 4^(k-1);
                for p = 1:block
                    for i = 1:2
                        tempY = 0;
                        for j = 1:c/2/block
                            XT = [XT;totalX+x];
                            YT = [YT;totalY+tempY+y];
                            if iscell
                               S = rdata.A12.U{k,(p-1)*c/block+(i-1)*(c/2/block)+j}+1i*idata.A12.U{k,(p-1)*c/block+(i-1)*(c/2/block)+j};
                            else
                               S = (squeeze(rdata.A12.U(k,(p-1)*c/block+(i-1)*(c/2/block)+j,:,:)+1i*idata.A12.U(k,(p-1)*c/block+(i-1)*(c/2/block)+j,:,:)));
                            end
                            ST = [ST;S(:)];
                            tempY = tempY + 2*r;
                            totalX = totalX + r;
                        end
                    end
                    totalY = totalY + tempY;
                end
                A12.U{k} = sparse(YT,XT,ST,c*r,c*r);    
            end
            
            New = struct('isLeaf',0,'sz',rdata.sz,'szsub',rdata.szsub,'A11',A11,'A12',A12,'A21',[],'A22',A22); 
            
        end
end

    function S = Snew(Sdatar,Sdatai)
        nns1 = size(Sdatar,1);
        if  nns1 > 4
            nns2 = size(Sdatar,2);
            nns2 = sqrt(nns2);
            nns = nns1*nns2;
            S = sparse(nns,nns);
            S(1:nns/4,1:nns/4) = Snew(Sdatar(1:nns1/4,:),Sdatai(1:nns1/4,:));
            S(nns/4+1:nns/2,nns/2+1:nns*3/4) = Snew(Sdatar(nns1/4+1:nns1/2,:),Sdatai(nns1/4+1:nns1/2,:));
            S(nns/2+1:nns*3/4,nns/4+1:nns/2) = Snew(Sdatar(nns1/2+1:nns1*3/4,:),Sdatai(nns1/2+1:nns1*3/4,:));
            S(nns*3/4+1:nns,nns*3/4+1:nns) = Snew(Sdatar(nns1*3/4+1:nns1,:),Sdatai(nns1*3/4+1:nns1,:));
        else
            nns2 = size(Sdatar,2);
            nns2 = sqrt(nns2);
            x = ones(nns2,1)*(1:nns2);
            x = x(:);
            y = repmat((1:nns2).',1,nns2);
            y = y(:);
            XT = x;
            YT = y;
            S1 = (squeeze(Sdatar(1,:)+1i*Sdatai(1,:)));
            %S1 = S1.';
            ST = S1(:);
            XT = [XT;2*nns2+x];
            YT = [YT;nns2+y];
            S1 = (squeeze(Sdatar(2,:)+1i*Sdatai(2,:)));
            %S1 = S1.';
            ST = [ST;S1(:)];
            XT = [XT;nns2+x];
            YT = [YT;2*nns2+y];
            S1 = (squeeze(Sdatar(3,:)+1i*Sdatai(3,:)));
            %S1 = S1.';
            ST = [ST;S1(:)];
            XT = [XT;3*nns2+x];
            YT = [YT;3*nns2+y];
            S1 = (squeeze(Sdatar(4,:)+1i*Sdatai(4,:)));
            %S1 = S1.';
            ST = [ST;S1(:)];
            
            S = sparse(YT,XT,ST,4*nns2,4*nns2);
        end
    end
     
                
end

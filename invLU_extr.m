function [LU_rl,LU_im] = invLU_extr(F,L_or_U)
LU_rl = F;
LU_im = F;
switch L_or_U
    case 'L'
        if  F.isLeaf
            LU_rl.A11 = real(F.A11(:));
            LU_rl.A22 = real(F.A22(:));
            LU_rl.A21 = real(F.A21(:));
            LU_im.A11 = imag(F.A11(:));
            LU_im.A22 = imag(F.A22(:));
            LU_im.A21 = imag(F.A21(:));
        else
            [LU_rl.A11,LU_im.A11] = invLU_extr(F.A11,'L');
            [LU_rl.A22,LU_im.A22] = invLU_extr(F.A22,'L');
            N = F.szsub;
            lvl = size(F.A21.U,1);
            m = numel(F.A21.data{1}(1,1).ri);
            r = numel(F.A21.data{1}(1,1).rsk);
            c = size(F.A21.V{1},1)/numel(F.A21.data{1}(1,1).rsk);
            r_s = size(F.A21.S,1)/4^(size(F.A21.U,1));
            LU_rl.A21.data = [N,lvl,m,r,c,r_s];
            LU_im.A21.data = LU_rl.A21.data;
            
            LU_rl.A21.V{1} = zeros(c,r*m);
            LU_im.A21.V{1} = zeros(c,r*m);
            totalR = 0;
            totalC = 0;
            for i = 1:1
                for j = 1:2
                    temp = 0;
                    for k = 1:c/2
                        LU_rl.A21.V{1}((i-1)*c+(j-1)*c/2+k,:) = ...
                            reshape(full(real(F.A21.V{1}(totalR+1:totalR+r,totalC+temp+1:totalC+temp+m))),1,r*m);
                        LU_im.A21.V{1}((i-1)*c+(j-1)*c/2+k,:) = ...
                            reshape(full(imag(F.A21.V{1}(totalR+1:totalR+r,totalC+temp+1:totalC+temp+m))),1,r*m);
                        temp = temp + m;
                        totalR = totalR + r;
                    end
                end
                totalC = totalC + m*c;
            end
            
            LU_rl.A21.U{1} = zeros(c,r*m);
            LU_im.A21.U{1} = zeros(c,r*m);
            totalR = 0;
            totalC = 0;
            for i = 1:1
                for j = 1:2
                    temp = 0;
                    for k = 1:c/2
                        LU_rl.A21.U{1}((i-1)*c+(j-1)*c/2+k,:) = ...
                            reshape(full(real(F.A21.U{1}(totalR+temp+1:totalR+temp+m,totalC+1:totalC+r))),1,r*m);
                        LU_im.A21.U{1}((i-1)*c+(j-1)*c/2+k,:) = ...
                            reshape(full(imag(F.A21.U{1}(totalR+temp+1:totalR+temp+m,totalC+1:totalC+r))),1,r*m);
                        temp = temp + m;
                        totalC = totalC + r;
                    end    
                end
                totalR = totalR + m*c;
            end

            for h = 2:lvl

                LU_rl.A21.U{h} = zeros(c,2*r*r);
                LU_im.A21.U{h} = zeros(c,r*2*r);
                LU_rl.A21.V{h} = zeros(c,2*r*r);
                LU_im.A21.V{h} = zeros(c,r*2*r);

                totalR = 0;
                totalC = 0;
                for i = 1:4^(h-1)
                    for j = 1:2
                        temp = 0;
                        for k = 1:c/4^(h-1)/2
                            LU_rl.A21.V{h}((i-1)*c/4^(h-1)+(j-1)*c/4^(h-1)/2+k,:) = ...                
                                reshape(full(real(F.A21.V{h}(totalR+1:totalR+r,totalC+temp+1:totalC+temp+2*r))),1,2*r^2);
                            LU_im.A21.V{h}((i-1)*c/4^(h-1)+(j-1)*c/4^(h-1)/2+k,:) = ...
                                reshape(full(imag(F.A21.V{h}(totalR+1:totalR+r,totalC+temp+1:totalC+temp+2*r))),1,2*r^2);
                            temp = temp + 2*r;
                            totalR = totalR + r;
                        end
                    end
                    totalC = totalC + 2*r*c/4^(h-1)/2;
                end

                totalR = 0;
                totalC = 0;
                for i = 1:4^(h-1)
                    for j = 1:2
                        temp = 0;
                        for k = 1:c/4^(h-1)/2
                            LU_rl.A21.U{h}((i-1)*c/4^(h-1)+(j-1)*c/4^(h-1)/2+k,:) = ...
                                reshape(full(real(F.A21.U{h}(totalR+temp+1:totalR+temp+2*r,totalC+1:totalC+r))),1,2*r^2);
                            LU_im.A21.U{h}((i-1)*c/4^(h-1)+(j-1)*c/4^(h-1)/2+k,:) = ...
                                reshape(full(imag(F.A21.U{h}(totalR+temp+1:totalR+temp+2*r,totalC+1:totalC+r))),1,2*r^2);
                            temp = temp + 2*r;
                            totalC = totalC + r;
                        end    
                    end
                    totalR = totalR + 2*r*c/4^(h-1)/2;
                end
            end
            [LU_rl.A21.S,LU_im.A21.S] = S_info(F.A21.S, 4^lvl, r_s);
        end
    case 'U'
        if  F.isLeaf
            LU_rl.A11 = real(F.A11(:));
            LU_rl.A22 = real(F.A22(:));
            LU_rl.A12 = real(F.A12(:));
            LU_im.A11 = imag(F.A11(:));
            LU_im.A22 = imag(F.A22(:));
            LU_im.A12 = imag(F.A12(:));
        else
            [LU_rl.A11,LU_im.A11] = invLU_extr(F.A11,'U');
            [LU_rl.A22,LU_im.A22] = invLU_extr(F.A22,'U');
            N = F.szsub;
            lvl = size(F.A12.U,1);
            m = numel(F.A12.data{1}(1,1).ri);
            r = numel(F.A12.data{1}(1,1).rsk);
            c = size(F.A12.V{1},1)/numel(F.A12.data{1}(1,1).rsk);
            r_s = size(F.A12.S,1)/4^(size(F.A12.U,1));
            LU_rl.A12.data = [N,lvl,m,r,c,r_s];
            LU_im.A12.data = LU_rl.A12.data;

            LU_rl.A12.V{1} = zeros(c,r*m);
            LU_im.A12.V{1} = zeros(c,r*m);
            totalR = 0;
            totalC = 0;
            for i = 1:1
                for j = 1:2
                    temp = 0;
                    for k = 1:c/2
                        LU_rl.A12.V{1}((i-1)*c+(j-1)*c/2+k,:) = ...
                            reshape(full(real(F.A12.V{1}(totalR+1:totalR+r,totalC+temp+1:totalC+temp+m))),1,r*m);
                        LU_im.A12.V{1}((i-1)*c+(j-1)*c/2+k,:) = ...
                            reshape(full(imag(F.A12.V{1}(totalR+1:totalR+r,totalC+temp+1:totalC+temp+m))),1,r*m);
                        temp = temp + m;
                        totalR = totalR + r;
                    end
                end
                totalC = totalC + m*c;
            end
            
            LU_rl.A12.U{1} = zeros(c,r*m);
            LU_im.A12.U{1} = zeros(c,r*m);
            totalR = 0;
            totalC = 0;
            for i = 1:1
                for j = 1:2
                    temp = 0;
                    for k = 1:c/2
                        LU_rl.A12.U{1}((i-1)*c+(j-1)*c/2+k,:) = ...
                            reshape(full(real(F.A12.U{1}(totalR+temp+1:totalR+temp+m,totalC+1:totalC+r))),1,r*m);
                        LU_im.A12.U{1}((i-1)*c+(j-1)*c/2+k,:) = ...
                            reshape(full(imag(F.A12.U{1}(totalR+temp+1:totalR+temp+m,totalC+1:totalC+r))),1,r*m);
                        temp = temp + m;
                        totalC = totalC + r;
                    end    
                end
                totalR = totalR + m*c;
            end

            for h = 2:lvl

                LU_rl.A12.U{h} = zeros(c,2*r*r);
                LU_im.A12.U{h} = zeros(c,r*2*r);
                LU_rl.A12.V{h} = zeros(c,2*r*r);
                LU_im.A12.V{h} = zeros(c,r*2*r);

                totalR = 0;
                totalC = 0;
                for i = 1:4^(h-1)
                    for j = 1:2
                        temp = 0;
                        for k = 1:c/4^(h-1)/2
                            LU_rl.A12.V{h}((i-1)*c/4^(h-1)+(j-1)*c/4^(h-1)/2+k,:) = ...                
                                reshape(full(real(F.A12.V{h}(totalR+1:totalR+r,totalC+temp+1:totalC+temp+2*r))),1,2*r^2);
                            LU_im.A12.V{h}((i-1)*c/4^(h-1)+(j-1)*c/4^(h-1)/2+k,:) = ...
                                reshape(full(imag(F.A12.V{h}(totalR+1:totalR+r,totalC+temp+1:totalC+temp+2*r))),1,2*r^2);
                            temp = temp + 2*r;
                            totalR = totalR + r;
                        end
                    end
                    totalC = totalC + 2*r*c/4^(h-1)/2;
                end

                totalR = 0;
                totalC = 0;
                for i = 1:4^(h-1)
                    for j = 1:2
                        temp = 0;
                        for k = 1:c/4^(h-1)/2
                            LU_rl.A12.U{h}((i-1)*c/4^(h-1)+(j-1)*c/4^(h-1)/2+k,:) = ...
                                reshape(full(real(F.A12.U{h}(totalR+temp+1:totalR+temp+2*r,totalC+1:totalC+r))),1,2*r^2);
                            LU_im.A12.U{h}((i-1)*c/4^(h-1)+(j-1)*c/4^(h-1)/2+k,:) = ...
                                reshape(full(imag(F.A12.U{h}(totalR+temp+1:totalR+temp+2*r,totalC+1:totalC+r))),1,2*r^2);
                            temp = temp + 2*r;
                            totalC = totalC + r;
                        end    
                    end
                    totalR = totalR + 2*r*c/4^(h-1)/2;
                end
            end
            [LU_rl.A12.S,LU_im.A12.S] = S_info(F.A12.S, 4^lvl, r_s);
        end
end
        
    function [S_rl,S_im] = S_info(S, num, min)
        S_rl = zeros(num,min*min);
        S_im = zeros(num,min*min);
        if  mod(num,4) ~= 0
            disp('num should be a multiple of 4!!')
            return
        end
        if  num > 4
            [S_rl(1:num/4,:),S_im(1:num/4,:)] = S_info(S(1:num*min/4,1:num*min/4), num/4, min);
            [S_rl(num/4+1:num/2,:),S_im(num/4+1:num/2,:)] = S_info(S(num*min/4+1:num*min/2,num*min/2+1:3*num*min/4), num/4, min);
            [S_rl(num/2+1:3*num/4,:),S_im(num/2+1:3*num/4,:)] = S_info(S(num*min/2+1:3*num*min/4,num*min/4+1:num*min/2), num/4, min);
            [S_rl(3*num/4+1:num,:),S_im(3*num/4+1:num,:)] = S_info(S(3*num*min/4+1:num*min,3*num*min/4+1:num*min), num/4, min);
         
        else
            S_rl(1,:) = reshape(full(real(S(1:min,1:min))),1,min^2);
            S_rl(2,:) = reshape(full(real(S(min+1:2*min,2*min+1:3*min))),1,min^2);
            S_rl(3,:) = reshape(full(real(S(2*min+1:3*min,min+1:2*min))),1,min^2);
            S_rl(4,:) = reshape(full(real(S(3*min+1:4*min,3*min+1:4*min))),1,min^2);
            S_im(1,:) = reshape(full(imag(S(1:min,1:min))),1,min^2);
            S_im(2,:) = reshape(full(imag(S(min+1:2*min,2*min+1:3*min))),1,min^2);
            S_im(3,:) = reshape(full(imag(S(2*min+1:3*min,min+1:2*min))),1,min^2);
            S_im(4,:) = reshape(full(imag(S(3*min+1:4*min,3*min+1:4*min))),1,min^2);
        end
            
    end
end
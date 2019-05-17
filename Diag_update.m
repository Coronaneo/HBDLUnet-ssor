function N = Diag_update(F,d)
N = F;
m = F.szsub;
if  F.isLeaf
    N.A11 = F.A11 - diag(diag(F.A11)) + diag(d(1:m));
    N.A22 = F.A22 - diag(diag(F.A22)) + diag(d(m+1:2*m));
else
    N.A11 = Diag_update(F.A11,d(1:m));
    N.A22 = Diag_update(F.A22,d(m+1:2*m));
end
end
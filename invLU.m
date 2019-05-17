function LU = invLU(F)
LU = F;
if  F.isLeaf
    LU.A11 = inv(F.A11);
    LU.A22 = inv(F.A22);
else
    LU.A11 = invLU(F.A11);
    LU.A22 = invLU(F.A22);
end
end
function dmm = dmap_facies(fac)
% Representative grain diameter per facies [mm] (manuscript Table 2).
dmm = zeros(numel(fac),1);
dmm(fac=="coarse") = 30; dmm(fac=="medium") = 12; dmm(fac=="fine") = 5;
end

% affect to cluster depending on minimal distance et return the distortion
% criterion
function [w_label,distortion] = minimal_distance(ai_distances)
distortion = 0 ;
[M,w_label] = min(ai_distances,[],2) ;
distortion = sum (M) ;
end
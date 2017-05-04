% kmeans function
function [ao_centroid,distortion,distortionCost,converge] = Kmeans( ai_K , ai_Data , ai_numIteration )

ao_centroid = get_initial_Centers(ai_K, ai_Data) ;
distortionCost = zeros(ai_numIteration) ;

% local copy
w_Data = ai_Data ;

% initialize labels
w_label = zeros(size(ai_Data,1)) ;

for w_iter = 1:ai_numIteration
    % 1 / Evaluate distance to each centroid
    w_distance = evaluate_distances(ao_centroid, w_Data) ;
    % 2 / Find the minimum value
    test_label = w_label ;
    [w_label,distortion] = minimal_distance(w_distance) ;
    distortionCost(w_iter) = distortion ;
    % disp(['The distortion criterion at step ' num2str(w_iter) ' is : ' num2str(distortion)])
    if test_label == w_label
        converge = w_iter ;
        break
    end
    % 3 / Update Centroid Value
    ao_centroid = update_centers(ai_K, w_Data, w_label) ;
end
end
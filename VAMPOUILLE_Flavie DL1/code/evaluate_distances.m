% evaluate distances between data and centers
function ao_distances = evaluate_distances( ai_centroid , ai_Data)
ao_distances = zeros (size(ai_Data,1),size(ai_centroid,1)) ;
for i = 1 : size(ai_Data,1)
    for k = 1 : size(ai_centroid,1)
        ao_distances(i,k) = norm(ai_Data(i,:) - ai_centroid(k,:))^2 ;
    end
end
end
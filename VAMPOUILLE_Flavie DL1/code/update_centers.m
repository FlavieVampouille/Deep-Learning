% update center at each iteration
function ao_centroid = update_centers(ai_K , ai_data, w_label)
ao_centroid = zeros (ai_K,size(ai_data,2)) ; n = zeros (ai_K,1) ;
for k = 1 : ai_K
    for i = 1 : size(ai_data,1)
            if w_label(i) == k
                n(k) = n(k) + 1 ;
                ao_centroid(k,:) = ao_centroid(k,:) + ai_data(i,:) ;
            end
    end
    ao_centroid(k,:) = ao_centroid(k,:) / n(k) ;
end
end
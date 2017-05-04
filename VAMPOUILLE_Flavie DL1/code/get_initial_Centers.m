% get initial centers randomly
function ao_centers = get_initial_Centers(ai_K, ai_Data)
w_index = randi(size(ai_Data,1),1,ai_K) ;
ao_centers = ai_Data(w_index,:) ;
end
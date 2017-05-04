function [ x , ph ] = block_gibbs ( x0 , W , L )

x = x0 ;

for l = 1:L
    
    % probability that 'position' of h will be +1 given x
    ph          = 1 ./ ( 1 + exp(-2*x*W) ) ; 
    % decide whether to set 'position' of h to +1 or -1
    rand1       = rand(1,length(ph)) ;
    h           = double(ph>rand1) - ( 1 - double(ph>rand1) ) ;
    
    % probability that 'position' of x will be +1 given h
    px          = 1 ./ ( 1 + exp(2*h*W') ) ;
    % decide whether to set 'position' of x to +1 or -1
    rand2       = rand(1,length(px)) ;
    x           = double(px>rand2) - ( 1 - double(px>rand2) ) ;
    
end

ph = 1 ./ ( 1 + exp(-2*x*W) ) ;

end
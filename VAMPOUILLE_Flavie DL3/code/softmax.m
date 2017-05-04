function g = softmax(z)
g = bsxfun(@rdivide,exp(z),sum(exp(z),2));
end

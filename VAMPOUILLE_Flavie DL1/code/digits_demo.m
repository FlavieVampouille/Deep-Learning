clear all

%%
[Dtrain,Dtest]  = load_digit7;

whos
[nsamples,ndimensions] = size(Dtrain);

%%
meanDigit =0;
for k=1:nsamples
    meanDigit = meanDigit + Dtrain(k,:)/nsamples;
end
%% simpler & faster
meanDigit = mean(Dtrain,1)';
meanImage = reshape(meanDigit,[28,28]);
figure,imshow(meanImage);

%% calcul covariance matrix
covDigits = 0;
DtrainMinusMean = Dtrain - repmat(meanDigit',nsamples,1);
for k=1:nsamples
    covDigits = covDigits + (DtrainMinusMean(k,:)'*DtrainMinusMean(k,:)) / (nsamples-1);
end

%% matlab fonction
covDigitsMatlab = cov(Dtrain);

%% Check if they are the same
sum ( sum ( abs(covDigitsMatlab - covDigits) ) )

%% make sure covDigitsMatlab = your covDigits
figure,imagesc(covDigitsMatlab); title('covDigitMatlab');
figure,imagesc(covDigits); title('covDigit');

%% get top-5 eigenvectors
[eigvec,eigvals] = eigs(covDigitsMatlab,5); % eigvec = matrix whose columns are the corresponding eigenvectors 
                                            % eigvals = diagonal matrix containing the 5 eigenvalues on the main diagonal

figure,
subplot(1,3,1); imshow(reshape(eigvec(:,1),[28,28]),[])
subplot(1,3,2); imshow(reshape(eigvec(:,2),[28,28]),[])
subplot(1,3,3); imshow(reshape(eigvec(:,2),[28,28]),[])

for basis_idx = [1:3]
    factors =[-2,0,2];
    figure,
    for k=1:3
        imshow(reshape(meanDigit + 2*factors(k)*eigvec(:,basis_idx),[28,28]))
        pause
    end
end




%%
%% 1.1 PCA
%%


%% 11 first eigenvectors and eigenvalues
[eigvec,eigvals] = eigs(covDigitsMatlab,11) ;

%% Number of images in Dtest
ntest = size(Dtest,1) ;

%% Centered Dtest matrix
DtestCentered = Dtest - repmat(meanDigit',ntest,1);

%% expansion coefficients : Cnd = <In,eigvec(d)> / norm(eigvec(d))^2
Cnd = zeros(ntest,11) ;
for d=1:11
    for n=1:ntest
        Cnd(n,d) = dot(DtestCentered(n,:)',eigvec(:,d)) / norm(eigvec(:,d))^2 ;
    end
end

%% Quality of the learned model
E = zeros(1,11) ;
for D=1:11
    for n=1:ntest
        S = meanDigit ;
        for d=1:D
            S = S + Cnd(n,d)*eigvec(:,d) ;
        end
        E(D) = E(D) + norm ( Dtest(n,:)' - S ) ;
    end
end

%% plot E(D)
figure ;
E = E(1:10); 
input = [1:1:10] ;
plot ( input , E )
title('Error of the PCA learned model') ;
xlabel('Dimensionality of the model') ;
ylabel('E(D)') ;




%% 
%% 1.2 K-means
%% 

%% number of centroids
w_K = 2 ;

%% number max of iterations
w_numIteration = 100 ;

%% return the w_K group centers, print criterion distortion at each step
[finalCentroid,distortion,distortionCost,converge] = Kmeans( w_K , Dtrain , w_numIteration ) ;

%% plot the ditortion at each step
input = (1:converge) ;
distortionCost = distortionCost(distortionCost~=0);
plot (input,distortionCost) ;
title(['distortion criterion at each step with convergence at step ' num2str(converge)]) ;

%% Run 10 times k-means functions with different initial centers and keep the best solution

bestDistortion = Inf(1) ;
for t = 1:10
    [finalCentroid,distortion,distortionCost,converge] = Kmeans( w_K , Dtrain , w_numIteration ) ;
    if distortion < bestDistortion
        bestDistortion = distortion ;
        bestCentroid = finalCentroid ;
        bestDistortionCost = distortionCost ;
        bestConverge = converge ;
    end
end

disp ( ['The best distortion on Dtrain for K = 2 is : ' num2str(bestDistortion)] )

input = (1:bestConverge) ;
bestDistortionCost = bestDistortionCost(bestDistortionCost~=0);
plot (input,bestDistortionCost) ;
title(['Distortion criterion at each step with convergence at step ' num2str(bestConverge)]) ;


%% resulting digit cluster

figure ;
for k = 1 : w_K 
    subplot(1,w_K,k) ;
    ClusterImage = reshape(bestCentroid(k,:),[28,28]) ;
    imshow(ClusterImage,[]) ;
    title (['Digit cluster for K = ', num2str(w_K), ' and k = ', num2str(k)] ) ;
end

%% Repeat the process for K = 3, 4, 5, 10, 50, 100

allCentroid = cell([6,1]) ;
distortionDtrain = [] ;
n = 1 ;

for w_K=[3,4,5,10,50,100]

bestDistortion = Inf(1) ;
for t = 1:10
    [finalCentroid,distortion,distortionCost,converge] = Kmeans( w_K , Dtrain , w_numIteration ) ;
    if distortion < bestDistortion
        bestDistortion = distortion ;
        bestCentroid = finalCentroid ;
        bestDistortionCost = distortionCost ;
        bestConverge = converge ;
    end
end

disp ( ['The best distortion on Dtrain for K = ' num2str(w_K) ' is : ' num2str(bestDistortion)] )

figure ;
input = (1:bestConverge) ;
bestDistortionCost = bestDistortionCost(bestDistortionCost~=0);
plot (input,bestDistortionCost) ;
title(['Distortion criterion at each step for K = ' num2str(w_K) ' with convergence at step ' num2str(bestConverge)]) ;

distortionDtrain = [distortionDtrain,bestDistortion] ;
allCentroid{n} = bestCentroid ;
n = n+1 ;

figure ;
for k = 1 : w_K 
    subplot(round(sqrt(w_K)),round(sqrt(w_K))+1,k) ;
    ClusterImage = reshape(bestCentroid(k,:),[28,28]) ;
    imshow(ClusterImage,[]) ;
end

end

%% plot the distortion cost on Dtrain 
image ;
input = [3,4,5,10,50,100] ;
plot (input,distortionDtrain) ;
title ('Distortion cost on Dtrain depending of the number of clusters.') ;
xlabel('K') ;
ylabel('Distortion criterion') ;

%% Report the distortion cost on the testing data

n = 1 ;
distortionDtest = [] ;
for w_K=[3,4,5,10,50,100]
    distances = evaluate_distances(allCentroid{n},Dtest) ;
    [w_label,distortion] = minimal_distance(distances) ;
    disp ( ['The distortion cost for K = ' num2str(w_K) ' clusters on Dtest is : ' num2str(distortion)] )
    distortionDtest = [distortionDtest,distortion] ;
    n = n+1 ;
end

image ;
input = [3,4,5,10,50,100] ;
plot (input,distortionDtest) ;
title ('Distortion cost on Dtest depending of the number of clusters.') ;
xlabel('K') ;
ylabel('Distortion criterion') ;

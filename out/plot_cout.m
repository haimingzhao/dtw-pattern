function plot_cout(seriesfile, dfile)

% row and column offset start from 0
S = csvread(seriesfile, 1, 0);
a = S(:, 1);
b = S(:, 2);
M = csvread(dfile);

figure;
subplot(5,5,[1,6,11,16]); plot(a); view(270,90)
subplot(5,5,22:25); plot(b);

subplot(5,5,[2:5, 7:10, 12:15, 17:20 ]);
imagesc(flipud(M));
colorbar();
set(gca, 'YDir', 'reverse'); 
title('Dynamic time warping');

end

% run it like
% matlab -nodesktop -nosplash -r plot_cout("../data/small.csv", "D.csv")
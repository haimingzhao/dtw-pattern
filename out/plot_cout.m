% a = dlmread('a.txt');
% b = dlmread('b.txt');
M = dlmread('D.txt');

figure;
% subplot(5,5,[1,6,11,16]); plot(a); view(270,90)
% subplot(5,5,22:25); plot(b);

subplot(5,5,[2:5, 7:10, 12:15, 17:20 ]);
imagesc(flipud(M));
colorbar();
set(gca, 'YDir', 'reverse'); 
title('Dynamic time warping');

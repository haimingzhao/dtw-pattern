function [a,b,C,D] = plot_cout(seriesfile, cfile, dfile)

% row and column offset start from 0
S = csvread(seriesfile, 1, 0);
a = S(:, 1);
b = S(:, 2);

% remove the 0s in the end for matlab csv reading to display the exact data
while a(end)==0
    a = a(1:end-1);
end 

while b(end)==0
    b = b(1:end-1);
end    

C = csvread(cfile);
D = csvread(dfile);

% plot C matrix
figure;
subplot(5,6,[1,7,13,19]); plot(a); axis tight; view(270,90)
subplot(5,6,26:29); plot(b); axis tight;

subplot(5,6,[2:6, 8:12, 14:18, 20:24 ]);
imagesc(flipud(C(:, 1:end-1)));
colorbar();
set(gca, 'YDir', 'reverse'); 
title(cfile);

% plot D matrix
figure;
subplot(5,5,[1,6,11,16]); plot(a); axis tight; view(270,90)
subplot(5,5,22:25); plot(b); axis tight;

subplot(5,5,[2:5, 7:10, 12:15, 17:20 ]);
imagesc(flipud(D(:, 1:end-1)));
set(gca, 'YDir', 'reverse');
title(dfile);


% run it like
% matlab -nodesktop -nosplash -r 
% plot_cout('../data/small.csv', 'C.csv', 'D.csv');

end
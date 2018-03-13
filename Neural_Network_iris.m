%This is the script that is used to implement the data
%First manually import the wine data, Then split it into X and Y
X1=irisdata(:,1:4);
Y1=irisdata(:,5);
%This is the place that used to generate the 
size(X1);
size(Y1);
%First open the neural network tool box
nftool
Training_percentage=[0.3 0.4 0.5 0.6 0.7 0.8 0.9]
Mean_squared_error=[0.052149 0.053261 0.059716 0.031799 0.031766 0.0060557 0.0068592]
plot(Training_percentage, Mean_squared_error);
title('Mean squared_error change with Training size')
xlabel('Training percentage') % x-axis label
ylabel('Mean squared error') % y-axis label
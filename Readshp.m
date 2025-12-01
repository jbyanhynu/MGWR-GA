function [px,py,x,y] = Readshp(path)
data=xlsread(path,1); %#ok<XLSRD> 
y=data(:,1); % house price
px=data(:,2);% x coordinate
py=data(:,3);
onesMatrix=ones(length(px),1);
xx=data(:,4:end);
xx=(xx-mean(xx))./std(xx);
y=(y-mean(y))./std(y);
x=[onesMatrix xx];

end


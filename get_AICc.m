function AICc = get_AICc(bw,px,py,x,y)
[row_px,col_px]=size(x);
um=eye(row_px);
A=eye(row_px*col_px);
Q=zeros(row_px*col_px,row_px);
proMatrix = zeros(row_px, row_px, col_px);
distance_matrix=pdist2([px,py],[px,py]);
for varN=1:col_px
    kernel=exp(-0.5*((distance_matrix/bw(varN)).^2));
    xx=x(:,varN);
    xT=xx'.*kernel;
    xtx=sum(xT.*(xx'),2);
    inv_xtx=1./xtx;
    xtx_inv_xt=inv_xtx.*xT;
    influ=xx.*xtx_inv_xt;
    repeatedSubMatrix=repmat(influ,1,col_px);
    startP=(varN-1)*row_px+1;
    endP=varN*row_px;
    repeatedSubMatrix(1:row_px,startP:endP)=um;
    A(startP:endP,:)=repeatedSubMatrix;
    Q(startP:endP,:)=influ;
end
% A=sparse(A);
% b=sparse(b);
% f = bicg(A,b);
RR=A\Q;
f=RR*y;

fity=zeros(row_px,1);
for k1=1:col_px
    startP=(k1-1)*row_px+1;
    endP=k1*row_px;
    proMatrix(:,:,k1) = RR(startP:endP, :);
    fity=fity+f(startP:endP,1);
end

tr_S=trace(sum(proMatrix,3));
list_resid=y-fity;
SSE=list_resid'*list_resid;
sigma_hat=sqrt(SSE/row_px);
AICc=2*row_px*log(sigma_hat)+row_px*log(2*pi)+row_px*((row_px+tr_S)/(row_px-2-tr_S));
end

